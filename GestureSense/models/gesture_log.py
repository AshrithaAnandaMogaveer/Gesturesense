"""
GestureLog model – records each detected gesture per user session.
"""

from datetime import datetime
from extensions import db


class GestureLog(db.Model):
    __tablename__ = "gesture_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    gesture_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "gesture_name": self.gesture_name,
            "confidence": round(self.confidence * 100, 1),
            "detected_at": self.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def __repr__(self) -> str:
        return f"<GestureLog {self.gesture_name} @ {self.confidence:.2f}>"
