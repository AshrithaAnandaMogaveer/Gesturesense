"""
Main public routes – landing page, contact.
"""

import re
from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app
from extensions import db
from models.contact_message import ContactMessage

main_bp = Blueprint("main", __name__)

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@main_bp.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@main_bp.route("/favicon.ico")
def favicon():
    """Serve favicon to prevent 404 on browser auto-request."""
    return send_from_directory(
        current_app.static_folder,
        "favicon.svg",
        mimetype="image/svg+xml",
    )


@main_bp.route("/contact", methods=["POST"])
def contact():
    """
    Handle contact form submission.
    Accepts both regular POST (form) and AJAX (JSON).
    Returns JSON so the frontend can show inline feedback without page reload.
    """
    # Support both form-encoded and JSON bodies
    if request.is_json:
        data    = request.get_json(silent=True) or {}
        name    = str(data.get("name",    "")).strip()
        email   = str(data.get("email",   "")).strip().lower()
        message = str(data.get("message", "")).strip()
    else:
        name    = request.form.get("name",    "").strip()
        email   = request.form.get("email",   "").strip().lower()
        message = request.form.get("message", "").strip()

    # ── Validation ─────────────────────────────────────────────────────────────
    errors = {}

    if not name:
        errors["name"] = "Name is required."
    elif len(name) < 2:
        errors["name"] = "Name must be at least 2 characters."
    elif len(name) > 100:
        errors["name"] = "Name must be 100 characters or fewer."

    if not email:
        errors["email"] = "Email address is required."
    elif not _EMAIL_RE.match(email):
        errors["email"] = "Please enter a valid email address."

    if not message:
        errors["message"] = "Message is required."
    elif len(message) < 10:
        errors["message"] = "Message must be at least 10 characters."
    elif len(message) > 2000:
        errors["message"] = "Message must be 2000 characters or fewer."

    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    # ── Save to database ────────────────────────────────────────────────────────
    try:
        msg = ContactMessage(
            name       = name,
            email      = email,
            message    = message,
            ip_address = request.remote_addr,
        )
        db.session.add(msg)
        db.session.commit()
    except Exception as e:
        return jsonify({
            "status":  "error",
            "message": "Failed to save your message. Please try again."
        }), 500

    return jsonify({
        "status":  "success",
        "message": f"Thank you, {name}! Your message has been received. We'll get back to you at {email} soon."
    })
