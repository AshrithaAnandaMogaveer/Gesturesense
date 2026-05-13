"""
Chat routes – Mistral 7B powered sign language assistant.

  GET  /chat/           – full-page chat UI
  POST /chat/message    – send a message, get AI response
  GET  /chat/status     – model loading status
"""

from flask import Blueprint, render_template, request, jsonify, session
from flask_login import login_required, current_user
from utils.llm_engine import chat, get_status

chat_bp = Blueprint("chat", __name__)

# Max conversation history kept in session (to avoid huge context)
MAX_HISTORY = 20


@chat_bp.route("/")
@login_required
def chat_page():
    """Render the full-page chat UI."""
    return render_template("chat.html")


@chat_bp.route("/status")
@login_required
def model_status():
    """Return current model loading status."""
    return jsonify(get_status())


@chat_bp.route("/message", methods=["POST"])
@login_required
def send_message():
    """
    Accept a user message, run it through Mistral 7B, return the response.
    Conversation history is stored in the Flask session.
    """
    data = request.get_json(silent=True) or {}
    user_msg = str(data.get("message", "")).strip()[:1000]   # cap at 1000 chars

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Load conversation history from session
    history = session.get("chat_history", [])

    # Append user message
    history.append({"role": "user", "content": user_msg})

    # Keep only last MAX_HISTORY messages to avoid context overflow
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    # Get AI response
    reply = chat(history)

    # Append assistant reply to history
    history.append({"role": "assistant", "content": reply})
    session["chat_history"] = history
    session.modified = True

    return jsonify({
        "reply":   reply,
        "history": len(history),
    })


@chat_bp.route("/clear", methods=["POST"])
@login_required
def clear_history():
    """Clear the conversation history."""
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})
