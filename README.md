# 🤟 Sign Language Recognition System

A real-time computer vision and machine learning-based system that recognizes static sign language hand gestures through a webcam and converts them into text.

The system uses MediaPipe Hands for hand landmark detection and a Random Forest classifier for gesture recognition.

---

## 📌 Project Overview

Communication between sign language users and people unfamiliar with sign language can be challenging.

This project aims to reduce this communication barrier by providing a real-time system capable of recognizing predefined hand gestures and converting them into readable text.

The system captures video from a webcam, detects the user's hand, extracts 21 hand landmarks, converts them into 63 numerical features, and uses a trained Random Forest machine learning model to predict the corresponding sign.

Recognized characters can be combined to form words and sentences. The system also provides word suggestions and text-to-speech functionality.

---

## ✨ Features

- Real-time webcam-based sign recognition
- Hand detection using MediaPipe
- 21 hand landmark detection
- 63 numerical landmark features
- Recognition of A-Z alphabet gestures
- Space gesture
- Delete gesture
- Real-time prediction confidence
- Top 3 prediction display
- Word formation
- Word suggestions
- Delete and clear functionality
- Text-to-speech support
- Interactive graphical user interface
- Model evaluation and confusion matrix
- Machine learning performance analysis

---

## 🧠 Technology Stack

### Programming Language

- Python

### Computer Vision

- OpenCV
- MediaPipe

### Machine Learning

- Scikit-learn
- Random Forest Classifier

### Data Processing

- NumPy
- Pandas

### Model Management

- Joblib

### Visualization

- Matplotlib

### Other Technologies

- JSON
- Git
- GitHub

---

## 🏗️ System Architecture

The overall workflow of the system is:

Webcam
↓
Video Frame Capture
↓
Hand Detection
↓
MediaPipe Hand Landmarks
↓
21 Hand Landmarks
↓
63 Numerical Features
↓
Random Forest Classifier
↓
Gesture Prediction
↓
Character Processing
↓
Word Formation
↓
Word Suggestions
↓
Recognized Text
↓
Optional Text-to-Speech

---

## 📊 Dataset

The final dataset contains:

| Property | Value |
|---|---:|
| Total Samples | 12,496 |
| Number of Classes | 28 |
| Number of Features | 63 |
| Training Samples | 9,996 |
| Testing Samples | 2,500 |

### Classes

The system recognizes:

- A-Z
- `space`
- `del`

Total:

**28 classes**

Each hand is represented using 21 landmarks.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

Therefore:

**21 × 3 = 63 features**

---

## 🤖 Machine Learning Model

The project uses a Random Forest Classifier.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make a final prediction.

The model was optimized using RandomizedSearchCV and evaluated using stratified cross-validation.

### Best Model Parameters

```text
n_estimators: 300
min_samples_split: 5
min_samples_leaf: 1
max_features: sqrt
max_depth: None
```

### Author
Alok Singh
