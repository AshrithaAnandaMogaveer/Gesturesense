"""
User model – stores account credentials.
Passwords are stored as bcrypt-style hashes via Werkzeug.
"""

from datetime import datetime
from flask_login import UserMixin
from extensions import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to gesture logs
    gesture_logs = db.relationship("GestureLog", backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        """Hash and store the password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


@login_manager.user_loader
def load_user(user_id: int):
    """Flask-Login callback to reload the user from the session.
    Uses db.session.get() which is compatible with SQLAlchemy 2.x.
    (User.query.get() was removed in SQLAlchemy 2.0)
    """
    return db.session.get(User, int(user_id))
