"""
LLM Engine – connects to LM Studio (or Ollama) running locally.

LM Studio exposes an OpenAI-compatible REST API at http://localhost:1234/v1
The model (Mistral 7B Instruct) is loaded and managed entirely by LM Studio.
GestureSense just sends HTTP requests — no model files, no RAM management here.

To use:
  1. Open LM Studio
  2. Load any model (Mistral 7B Instruct recommended)
  3. Start the local server  (Server tab → Start Server)
  4. Run GestureSense normally — it auto-connects
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# ── LM Studio connection config ────────────────────────────────────────────────
LM_STUDIO_BASE = "http://host.docker.internal:1234/v1"
MODELS_URL     = f"{LM_STUDIO_BASE}/models"
CHAT_URL       = f"{LM_STUDIO_BASE}/chat/completions"
TIMEOUT        = 120   # seconds — LM Studio can be slow on first token

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are GestureSense AI — an expert assistant specialising in:\n"
    "• Sign language (ASL, BSL, ISL, and universal hand gestures)\n"
    "• Hand gesture recognition and meanings\n"
    "• Computer vision (OpenCV, MediaPipe hand/face landmarks)\n"
    "• The GestureSense platform (Flask, Python, real-time webcam detection)\n"
    "• Facial expressions and body language\n\n"
    "Answer clearly and helpfully. For sign language questions, describe how to "
    "form the sign, its meaning, and cultural context. Keep answers concise but "
    "complete. If asked about something unrelated, politely redirect to your area "
    "of expertise."
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get(url: str) -> Optional[dict]:
    """Simple GET request, returns parsed JSON or None on error."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _post(url: str, payload: dict) -> Optional[dict]:
    """POST JSON payload, returns parsed JSON or None on error."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data    = data,
            headers = {"Content-Type": "application/json", "Accept": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        logger.error("[LLM] POST error: %s", e)
        return None
    except Exception as e:
        logger.error("[LLM] Unexpected error: %s", e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def get_status() -> dict:
    """
    Check if LM Studio is running and has a model loaded.
    Returns a status dict consumed by the /chat/status endpoint.
    """
    result = _get(MODELS_URL)

    if result is None:
        return {
            "status":  "error",
            "message": (
                "LM Studio is not running. "
                "Open LM Studio → load Mistral 7B → Start Server on port 1234."
            ),
        }

    models = result.get("data", [])
    if not models:
        return {
            "status":  "error",
            "message": "LM Studio is running but no model is loaded. Load a model in LM Studio first.",
        }

    model_id = models[0].get("id", "unknown")
    return {
        "status":   "ready",
        "message":  f"Connected to LM Studio · {model_id}",
        "model":    model_id,
        "models":   [m.get("id") for m in models],
    }


def chat(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Send a conversation to the locally running LM Studio model.

    messages : list of {"role": "user"|"assistant", "content": "..."}
    Returns  : assistant reply string
    """
    # Check LM Studio is up before sending
    status = get_status()
    if status["status"] != "ready":
        return (
            "⚠️ **LM Studio is not connected.**\n\n"
            "Please:\n"
            "1. Open **LM Studio**\n"
            "2. Load **Mistral 7B Instruct** (or any model)\n"
            "3. Go to the **Local Server** tab\n"
            "4. Click **Start Server** (port 1234)\n\n"
            "Then refresh this page and try again."
        )

    model_id = status.get("model", "mistral-7b-instruct-v0.2")

    # Mistral Instruct does NOT support the "system" role in LM Studio.
    # Inject the system prompt into the first user message instead.
    full_messages = []
    for i, msg in enumerate(messages):
        if i == 0 and msg.get("role") == "user":
            # Prepend system context to the first user message
            combined = SYSTEM_PROMPT + "\n\n" + msg["content"]
            full_messages.append({"role": "user", "content": combined})
        else:
            full_messages.append(msg)

    if not full_messages:
        full_messages = messages

    payload = {
        "model":       model_id,
        "messages":    full_messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.95,
        "stream":      False,
    }

    result = _post(CHAT_URL, payload)

    if result is None:
        return (
            "❌ No response from LM Studio. "
            "Make sure the server is running on port 1234 and a model is loaded."
        )

    try:
        reply = result["choices"][0]["message"]["content"].strip()
        return reply if reply else "I couldn't generate a response. Please try again."
    except (KeyError, IndexError) as e:
        logger.error("[LLM] Unexpected response format: %s | %s", e, result)
        return "❌ Unexpected response format from LM Studio."


# ── Compatibility stubs (no-op — LM Studio manages the model) ─────────────────

def load_model() -> bool:
    """No-op: LM Studio manages model loading."""
    return True


def start_background_load():
    """No-op: LM Studio manages model loading."""
    import threading
    return threading.Thread(target=lambda: None, daemon=True)
