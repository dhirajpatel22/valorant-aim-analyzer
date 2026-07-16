# Valorant Aim Analyzer

An AI-powered computer vision application that analyzes Valorant gameplay footage and provides personalized insights on crosshair placement, aiming mechanics, and reaction time.

> 🚧 **Project Status:** Early development. Features and documentation are actively being improved.

---

## Overview

Valorant Aim Analyzer is a computer vision project designed to analyze gameplay videos and extract meaningful aim statistics. The goal is to build an AI coaching tool that helps players improve by identifying patterns in their aiming behavior, crosshair placement, and reaction time.

This project is being developed as a portfolio project to explore:

- Computer Vision
- Object Detection
- Deep Learning
- Video Processing
- AI-Assisted Game Analytics

---

## Current Features

✅ **Player Detection**
- Uses YOLO-based object detection to identify player locations in gameplay footage.
- Generates bounding boxes around detected players.

✅ **Crosshair Detection**
- Detects the player's crosshair position in each frame.
- Tracks crosshair movement during gameplay.

---

## Features In Progress

🚧 **Head Detection**
- Training a custom computer vision model to identify enemy head locations.
- Uses detected head positions to evaluate:
  - Crosshair placement accuracy
  - Distance between crosshair and target head
  - Aim adjustment efficiency

🚧 **Aim Analysis**
- Measuring how quickly and accurately the player moves their crosshair onto targets.
- Analyzing flicks, tracking, and micro-adjustments.

---

## Future Features

- Kill detection and event tracking
- Reaction time measurement
- Headshot accuracy analysis
- Crosshair placement scoring
- Aim improvement recommendations
- Heatmaps showing crosshair movement
- Round-by-round performance breakdown
- AI-generated coaching feedback

---

## Technologies Used

- Python
- OpenCV
- YOLO (Ultralytics)
- PyTorch
- Computer Vision
- Machine Learning

---

## Project Structure

```text
valorant-aim-analyzer/
│
├── data/                       # Training data (images), download from roboflow
├── input/                      # Input gameplay footage
├── runs/                       # Trained YOLO model
│ └── detect/
│   └── valorant_coach/
│       └── enemy_model_v1/     # Enemy detection model
│            └── weights/
│               ├── best.pt     # Best-performing trained model
│                └── last.pt
│       └── head_model_v1/      # Head detection model
│            └── weights/
│               ├── best.pt     
│                └── last.pt
├── train.py                    # Model training script
├── main.py                     # Video processing pipeline
├── requirements.txt
├── yolo11n.pt                  # Pre-trained YOLO11 Nano model used for transfer learning
└── README.md   
```

## Disclaimer

This project is intended for educational and research purposes. It analyzes recorded gameplay footage and does not interact with the game client or provide in-game assistance.