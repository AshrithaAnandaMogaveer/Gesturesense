"""
ContactMessage model – stores every contact form submission.
"""

from datetime import datetime
from extensions import db


class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(120), nullable=False)
    message    = db.Column(db.Text, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read    = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<ContactMessage from {self.name} ({self.email})>"
