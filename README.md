````markdown
# GestureSense

A real-time Indian Sign Language (ISL) gesture recognition system built with Flask, OpenCV, MediaPipe, and machine learning. GestureSense detects hand gestures and facial expressions through a camera-based interface and provides AI-assisted interaction through LM Studio.

## Features

- Real-time ISL hand gesture recognition
- Machine learning-based gesture classification
- MediaPipe hand and face landmark detection
- Facial expression recognition
- Image-based gesture processing
- AI-assisted interaction using LM Studio
- Web-based interface built with Flask
- Docker support for containerized deployment
- Jenkins support for CI/CD automation
- Session-based gesture detection and confidence tracking

## Technology Stack

- **Backend:** Flask
- **Computer Vision:** OpenCV
- **Landmark Detection:** MediaPipe
- **Machine Learning:** Scikit-learn
- **Model:** SVM with StandardScaler
- **Data Processing:** NumPy, SciPy
- **Model Serialization:** Joblib / Pickle
- **AI Assistant:** LM Studio
- **Containerization:** Docker
- **CI/CD:** Jenkins
- **Frontend:** HTML, CSS, JavaScript

## Machine Learning

The ISL recognition system uses MediaPipe hand landmarks to generate a 73-dimensional feature vector consisting of:

- 63 normalized hand landmark coordinates
- 5 finger-state features
- 5 fingertip/inter-finger distance features

The trained model uses a Scikit-learn pipeline:

```text
StandardScaler
      ↓
SVM (SVC)
      ↓
114 ISL Classes
````

The trained model and label mapping are stored in:

```text
models_cache/isl_model.pkl
models_cache/isl_labels.pkl
```

## Model Performance

Current model evaluation:

| Metric                           |         Result |
| -------------------------------- | -------------: |
| Classes                          |            114 |
| Features                         |             73 |
| Training Samples                 |          3,566 |
| 5-Fold Cross-Validation Accuracy | 77.14% ± 1.34% |
| Test Accuracy                    |         78.71% |
| Macro F1-Score                   |           0.78 |
| Weighted F1-Score                |           0.78 |

Model performance may vary depending on lighting, camera position, hand orientation, background, and signing conditions.

## Project Structure

```text
GestureSense/
├── app.py
├── train_isl_model.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
├── README.md
│
├── utils/
│   ├── gesture_engine.py
│   └── isl_classifier.py
│
├── templates/
├── static/
│
├── models_cache/
│   ├── isl_model.pkl
│   └── isl_labels.pkl
│
└── ISL_CSLRT_Corpus/
```

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd GestureSense
```

### 2. Create a Python Virtual Environment

Python 3.11 is recommended.

```bash
python -m venv training_venv
```

Activate it on Windows:

```powershell
.\training_venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file for environment-specific configuration.

Do not commit `.env` or other credentials/secrets to the repository.

For LM Studio, make sure the local model server is running and configured according to the application's environment settings.

## Running the Application

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Training the ISL Model

Place the required dataset in the expected project directory and run:

```bash
python train_isl_model.py
```

After successful training, the model files are generated in:

```text
models_cache/
```

```text
isl_model.pkl
isl_labels.pkl
```

The application automatically loads the trained model when available.

## Docker

Build the Docker image:

```bash
docker build -t gesturesense .
```

Run the container:

```bash
docker run -p 5000:5000 gesturesense
```

Open:

```text
http://localhost:5000
```

## CI/CD

The project includes a Jenkins pipeline for automated build and deployment workflows.

The Jenkins configuration is provided in:

```text
Jenkinsfile
```

## Dataset

The project uses the ISL-annotated gesture dataset available in the project environment for model training.

Dataset files may be subject to their own licensing and redistribution terms. Verify the dataset's license before redistributing it through a public repository.

## Important Notes

* Model performance depends on the quality and diversity of training data.
* Clear hand visibility improves recognition reliability.
* The trained model should be evaluated on data that is separate from the training data.
* Large datasets and trained model files may be excluded from GitHub depending on repository and licensing requirements.
* Keep API keys, credentials, and environment-specific configuration out of source control.

## Future Improvements

* Expand the number of supported ISL gestures
* Increase real-world training data diversity
* Improve recognition of visually similar gestures
* Improve temporal recognition for dynamic signs
* Enhance model evaluation and confusion-matrix analysis
* Improve real-time confidence calibration
* Expand deployment and CI/CD automation

## License

This project does not currently specify a license.

If this project is intended for public distribution, add an appropriate open-source license after confirming the licensing requirements for the source code and dataset.

## Author

**GestureSense Project**

---

If you find this project useful, consider giving the repository a ⭐.

```
```
