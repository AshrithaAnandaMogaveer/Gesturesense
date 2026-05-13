# GestureSense – AI Hand Gesture Recognition System

> Real-time AI-powered hand gesture recognition platform built with Python, Flask, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## Features

- **Real-time gesture detection** at 30 fps via webcam
- **10+ gesture classes**: Palm Open, Fist, Thumbs Up/Down, Victory, OK Sign, Point, Rock On, Call Me, Three, Four
- **Secure authentication**: register, login, logout with hashed passwords
- **Gesture history**: every detection is logged to SQLite per user
- **Futuristic dark UI**: glassmorphism, neon gradients, smooth animations
- **Docker + Jenkins CI/CD** ready

---

## Project Structure

```
GestureSense/
├── app.py                  # Application factory
├── extensions.py           # Shared Flask extensions (db, login_manager)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service orchestration
├── Jenkinsfile             # CI/CD pipeline
├── .env.example            # Environment variable template
│
├── models/
│   ├── user.py             # User ORM model
│   └── gesture_log.py      # GestureLog ORM model
│
├── routes/
│   ├── auth.py             # /auth/register, /auth/login, /auth/logout
│   ├── main.py             # / (landing), /contact
│   └── gesture.py          # /gesture/processing, /gesture/video_feed, etc.
│
├── utils/
│   └── gesture_engine.py   # OpenCV + MediaPipe engine + gesture classifier
│
├── static/
│   ├── css/custom.css      # CSS overrides
│   └── js/main.js          # Particle background + global utilities
│
└── templates/
    ├── base.html           # Master layout (navbar, footer, flash messages)
    ├── index.html          # Landing page
    ├── gesture_processing.html  # Webcam + detection UI
    └── auth/
        ├── login.html
        └── register.html
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd GestureSense
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY
```

### 3. Run

```bash
python app.py
```

Open http://localhost:5000

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t gesturesense .
docker run -p 5000:5000 gesturesense
```

---

## Supported Gestures

| Gesture | Description |
|---------|-------------|
| Palm Open | All 5 fingers extended |
| Fist | All fingers closed |
| Thumbs Up | Thumb extended, others closed |
| Thumbs Down | Thumb pointing down |
| Victory ✌ | Index + middle extended |
| OK Sign 👌 | Thumb + index tips touching |
| Point | Index finger only |
| Rock On 🤘 | Index + pinky extended |
| Call Me 🤙 | Thumb + pinky extended |
| Three | Index + middle + ring |
| Four | All except thumb |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| AI/CV | OpenCV 4.10, MediaPipe 0.10 |
| Database | SQLite via SQLAlchemy |
| Auth | Flask-Login, Werkzeug password hashing |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| DevOps | Docker, Jenkins |

---

## License

MIT © 2024 GestureSense
