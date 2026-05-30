# Motion-Capture-Sign-Language-Translation

> Real-time Sign Language Translation using Motion Capture, Computer Vision, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-HandTracking-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-red)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

# 📖 Overview

Motion-Capture-Sign-Language-Translation is a Data Science and Artificial Intelligence project that aims to translate sign language gestures into readable text and speech in real time.

The system utilizes webcam input, hand landmark extraction, machine learning classification, and visualization dashboards to provide communication assistance for sign language users.

---

# 🎯 Objectives

* Detect hand movements using webcam
* Extract hand landmarks
* Recognize sign language gestures
* Convert gestures into text
* Convert text into speech
* Provide realtime prediction

---

# 👥 Team Members

| Name                  | Role                                      |
| --------------------- | ----------------------------------------- |
| Afllah Abdi           | Computer Vision Engineer                  |
| Farhan Abdurrahman    | Machine Learning Engineer                 |
| Andika Candra         | Backend & Deployment Engineer             |
| Ahmad Nizar Rusdiawan | Frontend, Documentation & Project Manager |

---

# 🏗 System Architecture

User Gesture
↓
Webcam Input
↓
MediaPipe Hand Tracking
↓
Landmark Extraction
↓
Feature Engineering
↓
Machine Learning Model
↓
Gesture Prediction
↓
Text Output
↓
Speech Output

---

# 🚀 Main Features

## Motion Capture

Realtime hand movement detection using webcam.

## Hand Landmark Detection

Extract hand keypoints using MediaPipe.

## Gesture Recognition

Classify sign language gestures using machine learning models.

## Translation System

Convert recognized gestures into readable text.

## Text-to-Speech

Generate audio output automatically.

## Dashboard Visualization

Display realtime prediction results.

---

# 🧠 Technology Stack

## Computer Vision

* OpenCV
* MediaPipe

## Machine Learning

* Scikit-Learn
* TensorFlow
* PyTorch

## Backend

* FastAPI
* Flask

## Frontend

* Streamlit

## Text To Speech

* gTTS
* pyttsx3

---

# 📂 Repository Structure

```bash
Motion-Capture-Sign-Language-Translation/

├── dataset/

├── notebooks/

├── src/
│   ├── capture.py      # Hand tracking module (MediaPipe)
│   ├── collect_data.py # Data collection with quality validation
│   ├── preprocess.py   # Feature normalization
│   ├── train.py        # Model training
│   └── predict.py      # Gesture prediction

├── models/

├── api/

├── dashboard/

├── docs/

├── README.md
└── requirements.txt
```

---

# 🖐️ Hand Tracking Module (capture.py)

Real-time hand tracking menggunakan MediaPipe untuk ekstraksi 21 hand landmarks.

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| Webcam Integration | Akses webcam dengan OpenCV |
| MediaPipe Hands | Real-time hand detection |
| Landmark Extraction | 63 features per hand (21 landmarks × 3 coordinates) |
| Visualization | Skeleton + landmark points + bounding boxes |
| Debug Info | FPS counter + hand count display |

## Parameter MediaPipe

```python
static_image_mode=False      # Real-time tracking mode
max_num_hands=2             # Maximum 2 hands
min_detection_confidence=0.7
min_tracking_confidence=0.7
```

## Penggunaan

```bash
python -m src.capture
```

## Keyboard Controls

| Key | Fungsi |
|-----|--------|
| `q` | Quit aplikasi |
| `s` | Export landmarks ke console |
| `h` | Toggle hand labels (Left/Right) |
| `b` | Toggle bounding boxes |

## Output Format

Fungsi `extract_landmarks()` menghasilkan:
```python
[x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]  # 63 features per hand
```

## Integrasi

Modul ini digunakan oleh:
- `preprocess.py` - Data normalization
- `train.py` - Model training
- `predict.py` - Gesture prediction

# 📊 Dataset Plan

Initial gesture classes:

* Halo
* Makan
* Minum
* Tolong
* Terima Kasih

Dataset source:

* Custom Dataset (Webcam Recording)
* Public Dataset (ASL / Kaggle)

---

# 📈 Development Roadmap

## Phase 1 — MVP

* [x] Hand Detection
* [x] Landmark Extraction
* [ ] Dataset Collection
* [ ] Basic Classification
* [ ] Text Output

## Phase 2 — Intermediate

* [x] Multi-Hand Detection
* [ ] Text-to-Speech
* [ ] Confidence Score
* [ ] Prediction History

## Phase 3 — Advanced

* [ ] Sentence Builder
* [ ] Sequence Detection
* [ ] Realtime Sentence Prediction
* [ ] Indonesian Sign Language Support

---

# 📅 Daily Progress Log

## Day 1 — Project Initialization

Date:

Progress:

* Repository created
* Team formed
* Brainstorming completed
* Initial documentation completed

Challenges:

* None

Next Target:

* Dataset planning

---

## Day 2

Date:

## Progress:

## Challenges:

## Next Target:

---

## Day 3

Date:

## Progress:

## Challenges:

## Next Target:

---

## Day 4

Date:

## Progress:

## Challenges:

## Next Target:

---

## Day 5

Date:

## Progress:

## Challenges:

## Next Target:

---

# 📌 Current Progress

| Task               | Status |
| ------------------ | ------ |
| Brainstorming      | ✅      |
| Team Formation     | ✅      |
| Project Planning   | ✅      |
| Hand Tracking      | ✅      |
| Dataset Collection | ⬜      |
| Data Preprocessing | ⬜      |
| Model Training     | ⬜      |
| Testing            | ⬜      |
| Deployment         | ⬜      |
| Final Presentation | ⬜      |

---

# 🧪 Planned Models

Baseline Models:

* KNN
* SVM
* Random Forest

Advanced Models:

* CNN
* LSTM
* CNN + LSTM
* Transformer

---

# ⚠️ Challenges

Potential issues:

* Similar gestures
* Lighting variation
* Small dataset
* Overfitting
* Camera angle
* Realtime latency

---

# 🔮 Future Improvements

* Mobile Application
* Edge AI Deployment
* Realtime Subtitle Generator
* Multilingual Translation
* Indonesian Sign Language Support
* IoT Integration

---

# 📄 License

MIT License

---

# 👨‍💻 Developed By

Team Motion Capture Sign Language Translation

* Afllah Abdi
* Farhan Abdurrahman
* Andika Candra
* Ahmad Nizar Rusdiawan

Universitas Muhammadiyah Malang

2026
