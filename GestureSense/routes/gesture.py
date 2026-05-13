"""
Gesture processing routes – complete production API.

Endpoints:
  GET  /gesture/processing   – AI dashboard page
  GET  /gesture/video_feed   – MJPEG stream
  GET  /gesture/current      – latest gesture JSON
  POST /gesture/log          – persist a detection
  GET  /gesture/history      – last 50 logs for current user
  GET  /gesture/stats        – session + DB statistics
  POST /gesture/analyze_image – analyse uploaded image for gestures
"""

from datetime import datetime
from flask import Blueprint, render_template, Response, request, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from extensions import db
from models.gesture_log import GestureLog
from utils.gesture_engine import GestureEngine, process_image

gesture_bp = Blueprint("gesture", __name__)

# ── Single shared engine (daemon thread inside) ────────────────────────────────
_engine = GestureEngine()


# ── Pages ──────────────────────────────────────────────────────────────────────

@gesture_bp.route("/processing")
@login_required
def processing():
    """Render the full AI gesture dashboard."""
    return render_template("gesture_processing.html")


# ── Streaming ──────────────────────────────────────────────────────────────────

@gesture_bp.route("/video_feed")
@login_required
def video_feed():
    """
    MJPEG stream consumed by the <img> tag in the UI.
    The engine's background thread handles the webcam; this just
    reads from the shared frame buffer.
    """
    return Response(
        _engine.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── JSON APIs ──────────────────────────────────────────────────────────────────

@gesture_bp.route("/current")
@login_required
def current_gesture():
    """Return the latest detected hand gesture."""
    name, conf = _engine.get_latest()
    return jsonify({
        "gesture":    name,
        "confidence": round(conf * 100, 1),
        "timestamp":  datetime.utcnow().strftime("%H:%M:%S"),
        "has_hand":   name != "No Hand Detected",
    })


@gesture_bp.route("/current_face")
@login_required
def current_face():
    """Return the latest detected face expression."""
    expr, conf, detected = _engine.get_latest_face()
    return jsonify({
        "expression": expr,
        "confidence": round(conf * 100, 1),
        "detected":   detected,
        "timestamp":  datetime.utcnow().strftime("%H:%M:%S"),
    })


@gesture_bp.route("/log", methods=["POST"])
@login_required
def log_gesture():
    """Persist a detected gesture to the database."""
    data         = request.get_json(silent=True) or {}
    gesture_name = str(data.get("gesture", "Unknown"))[:100]
    confidence   = float(data.get("confidence", 0.0))

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    log = GestureLog(
        user_id=current_user.id,
        gesture_name=gesture_name,
        confidence=confidence,
    )
    db.session.add(log)
    db.session.commit()
    return jsonify({"status": "ok", "id": log.id})


@gesture_bp.route("/history")
@login_required
def history():
    """Return the last 50 gesture detections for the current user."""
    logs = (
        GestureLog.query
        .filter_by(user_id=current_user.id)
        .order_by(GestureLog.detected_at.desc())
        .limit(50)
        .all()
    )
    return jsonify([l.to_dict() for l in logs])


@gesture_bp.route("/stats")
@login_required
def stats():
    """
    Return combined statistics:
      - Live engine stats (FPS, session duration, hand count)
      - DB stats (total logs, most common gesture, avg confidence)
    """
    engine_stats = _engine.get_stats()

    # DB aggregates for this user
    total_db = db.session.query(func.count(GestureLog.id))\
        .filter_by(user_id=current_user.id).scalar() or 0

    avg_conf_db = db.session.query(func.avg(GestureLog.confidence))\
        .filter_by(user_id=current_user.id).scalar()
    avg_conf_db = round((avg_conf_db or 0) * 100, 1)

    # Most common gesture (all time for this user)
    top_row = (
        db.session.query(GestureLog.gesture_name, func.count(GestureLog.id).label("cnt"))
        .filter_by(user_id=current_user.id)
        .group_by(GestureLog.gesture_name)
        .order_by(func.count(GestureLog.id).desc())
        .first()
    )
    top_gesture = top_row[0] if top_row else "—"

    return jsonify({
        **engine_stats,
        "total_db":       total_db,
        "avg_conf_db":    avg_conf_db,
        "top_gesture":    top_gesture,
    })


# ── Image analysis ─────────────────────────────────────────────────────────────

@gesture_bp.route("/analyze_image", methods=["POST"])
@login_required
def analyze_image():
    """
    Accept an uploaded image file, run MediaPipe hand detection,
    classify all detected gestures, and return:
      - list of gestures with confidence
      - base64-encoded annotated image
    Also auto-logs each detected gesture to the DB.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if file.content_type not in allowed:
        return jsonify({"error": f"Unsupported file type: {file.content_type}"}), 400

    # Read bytes (max 10 MB)
    image_bytes = file.read(10 * 1024 * 1024)
    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file"}), 400

    # Process
    result = process_image(image_bytes)

    if result.get("error"):
        return jsonify(result), 422

    # Auto-log each detected gesture to DB
    for g in result.get("gestures", []):
        if g["confidence"] > 50:
            log = GestureLog(
                user_id=current_user.id,
                gesture_name=g["gesture"],
                confidence=g["confidence"] / 100.0,
            )
            db.session.add(log)
    if result.get("gestures"):
        db.session.commit()

    return jsonify(result)
