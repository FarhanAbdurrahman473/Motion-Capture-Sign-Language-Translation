# Motion Capture Sign Language Translation

## Overview

A computer vision and machine learning project that captures human body and hand movements using motion capture techniques and translates sign language into text in real time.

## Planned Features

* **Motion capture using webcam** — Real-time video capture for gesture recording
* **Hand and body landmark extraction** — MediaPipe-powered pose and hand keypoint detection
* **Dataset collection pipeline** — Structured storage and labeling of motion sequences
* **Data preprocessing** — Normalization, augmentation, and sequence alignment
* **Deep learning model training** — LSTM/Transformer architectures for sequence classification
* **Real-time sign language prediction** — Live inference with confidence scoring
* **REST API service** — FastAPI backend for programmatic access
* **Interactive dashboard** — Streamlit UI for visualization and control

## Repository Structure

```
Motion-Capture-Sign-Language-Translation/
├── dataset/           # Raw motion capture recordings and labels
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/               # Core Python modules
│   ├── capture.py     # Webcam capture and landmark extraction
│   ├── preprocess.py  # Data loading and augmentation
│   ├── train.py       # Model definition and training
│   └── predict.py     # Inference pipeline
├── models/            # Trained model checkpoints
├── api/               # FastAPI service code
├── dashboard/         # Streamlit dashboard code
├── docs/              # Project documentation
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## Roadmap

| Phase | Description |
|-------|-------------|
| **Phase 1** | Dataset Collection — Build capture pipeline and record initial dataset |
| **Phase 2** | Preprocessing — Normalize landmarks, align sequences, apply augmentations |
| **Phase 3** | Model Training — Design and train LSTM/Transformer classifier |
| **Phase 4** | Real-Time Inference — Optimize for low-latency prediction |
| **Phase 5** | API Development — Expose model via FastAPI endpoints |
| **Phase 6** | Dashboard Deployment — Streamlit UI for live demo and monitoring |
