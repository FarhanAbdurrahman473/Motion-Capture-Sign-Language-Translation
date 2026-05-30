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

---

# 🔧 Preprocessing Module (preprocess.py)

Pipeline preprocessing untuk mengkonversi raw landmarks menjadi normalized feature vectors.

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| Landmark Validation | Validasi 63 features, NaN check |
| Coordinate Normalization | Translation + scale invariance |
| Label Encoding | Konversi gesture → angka (0-4) |
| Dataset Builder | Load → Validate → Normalize → Export |
| Data Augmentation | Noise injection, scaling |

## Normalization Method

```python
# 1. Translation (wrist as origin)
x_norm = x - wrist_x
y_norm = y - wrist_y
z_norm = z - wrist_z

# 2. Scale normalization
landmarks = landmarks / max(abs(landmarks))
```

## Label Encoding

| Gesture | Label |
|---------|-------|
| Halo | 0 |
| Makan | 1 |
| Minum | 2 |
| Tolong | 3 |
| TerimaKasih | 4 |

## Penggunaan

```bash
# Run complete preprocessing pipeline
python -m src.preprocess

# Load existing processed dataset
python -m src.preprocess --load-only

# Skip normalization
python -m src.preprocess --no-normalize

# Custom directories
python -m src.preprocess --dataset dataset --output dataset/processed
```

## Output Files

```
dataset/processed/
├── X.npy              # Feature matrix (n_samples, 63)
├── y.npy              # Labels array (n_samples,)
├── dataset_info.json  # Metadata
└── dataset.csv        # Human-readable format
```

## Main Functions

| Function | Deskripsi |
|----------|-----------|
| `validate_landmarks()` | Validasi 63 features, NaN, range |
| `normalize_landmarks()` | Wrist-centered + scale normalization |
| `encode_label()` | Konversi gesture → integer |
| `build_dataset()` | Pipeline lengkap load → normalize |
| `save_dataset()` | Export ke .npy files |
| `print_dataset_summary()` | Statistics + visualization |

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
* [x] Data Preprocessing
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
| Data Preprocessing | ✅      |
| Dataset Collection | ⬜      |
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
