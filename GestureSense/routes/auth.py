"""
Authentication routes – register, login, logout.
Validation errors are returned as field-level dicts so the template
can render inline messages under each input without losing form values.
"""

import re
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models.user import User

auth_bp = Blueprint("auth", __name__)

# Simple email regex (good enough for server-side guard)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Username: letters, digits, underscores, hyphens only
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ── Register ───────────────────────────────────────────────────────────────────
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    # Field errors dict  {field_name: "error message"}
    errors = {}
    # Preserve submitted values so the form re-fills on error
    form_data = {"username": "", "email": ""}

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        form_data["username"] = username
        form_data["email"]    = email

        # ── Field-level validation ─────────────────────────────────────────
        if not username:
            errors["username"] = "Username is required."
        elif len(username) < 3:
            errors["username"] = "Username must be at least 3 characters."
        elif len(username) > 30:
            errors["username"] = "Username must be 30 characters or fewer."
        elif not _USERNAME_RE.match(username):
            errors["username"] = "Only letters, numbers, _ and - are allowed."
        elif User.query.filter_by(username=username).first():
            errors["username"] = "This username is already taken."

        if not email:
            errors["email"] = "Email address is required."
        elif not _EMAIL_RE.match(email):
            errors["email"] = "Please enter a valid email address."
        elif User.query.filter_by(email=email).first():
            errors["email"] = "An account with this email already exists."

        if not password:
            errors["password"] = "Password is required."
        elif len(password) < 6:
            errors["password"] = "Password must be at least 6 characters."
        elif len(password) > 128:
            errors["password"] = "Password must be 128 characters or fewer."

        if not confirm:
            errors["confirm_password"] = "Please confirm your password."
        elif password and confirm != password:
            errors["confirm_password"] = "Passwords do not match."

        # ── If no errors, create the user and log them in immediately ────────
        if not errors:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            # Auto-login right after registration – no need to sign in again
            login_user(user, remember=True)
            flash(f"Welcome to GestureSense, {user.username}! Your account is ready.", "success")
            return redirect(url_for("main.index"))

    return render_template(
        "auth/register.html",
        errors=errors,
        form_data=form_data,
    )


# ── Login ──────────────────────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    errors    = {}
    form_data = {"username": ""}

    if request.method == "POST":
        username_or_email = request.form.get("username", "").strip()
        password          = request.form.get("password", "")
        remember          = bool(request.form.get("remember"))

        form_data["username"] = username_or_email

        # ── Field-level validation ─────────────────────────────────────────
        if not username_or_email:
            errors["username"] = "Username or email is required."

        if not password:
            errors["password"] = "Password is required."

        # ── Credential check ───────────────────────────────────────────────
        if not errors:
            # Try matching by username first, then by email
            user = User.query.filter_by(username=username_or_email).first()
            if not user:
                user = User.query.filter_by(email=username_or_email.lower()).first()

            if user and user.check_password(password):
                login_user(user, remember=remember)
                next_page = request.args.get("next")
                flash(f"Welcome back, {user.username}!", "success")
                return redirect(next_page or url_for("main.index"))
            else:
                errors["credentials"] = "Invalid username/email or password. Please try again."

    return render_template(
        "auth/login.html",
        errors=errors,
        form_data=form_data,
    )


# ── Logout ─────────────────────────────────────────────────────────────────────
@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("main.index"))
