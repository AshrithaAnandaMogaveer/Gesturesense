"""
GestureSense – AI Hand Gesture Recognition System
Main application entry point.
"""

import os
from flask import Flask
from extensions import db, login_manager
from routes.auth import auth_bp
from routes.main import main_bp
from routes.gesture import gesture_bp
from routes.chat import chat_bp
from dotenv import load_dotenv

load_dotenv()


def create_app():
    """Application factory pattern."""
    app = Flask(__name__)

    # ── Configuration ──────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gestures3ns3-s3cr3t-k3y-2024")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///database.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

    # ── Session / cookie settings (required for Flask-Login on localhost) ──────
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"]   = False   # True only in production HTTPS
    app.config["REMEMBER_COOKIE_HTTPONLY"] = True
    app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"
    app.config["WTF_CSRF_ENABLED"]         = False  # No WTForms CSRF tokens in use

    # ── Extensions ─────────────────────────────────────────────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    # ── Blueprints ─────────────────────────────────────────────────────────────
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(gesture_bp, url_prefix="/gesture")
    app.register_blueprint(chat_bp, url_prefix="/chat")

    # ── Database initialisation ────────────────────────────────────────────────
    with app.app_context():
        db.create_all()

    # ── Start Mistral 7B loading in background ─────────────────────────────────
    from utils.llm_engine import start_background_load
    start_background_load()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
