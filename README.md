# Custom-Face-Recognition-System

## Overview
Real-time face recognition using a **custom dataset of 1246 images** across 101 individuals (5 angles per person). Features are extracted using **MobileNetV2** and classified via **SVM**. Supports **live webcam recognition** with unknown face detection using OpenCV.

---

## Features
- Load and preprocess images from a custom dataset.
- Encode labels automatically from folder names.
- Extract deep features using **MobileNetV2** pretrained on ImageNet.
- Normalize features with `StandardScaler`.
- Train an **SVM classifier** for face recognition.
- Real-time face detection with OpenCV Haar cascades.
- Unknown face detection for faces not in the dataset.

---

## Dataset Structure
Face/ # Root folder
├── Person1/ # Folder name used as label
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Person2/
│ ├── image1.jpg
│ └── ...
└── ...

- Each subfolder contains images of one person.
- Folder names are automatically converted into labels.

---

## Technologies
- Python 3.x
- OpenCV
- TensorFlow / Keras
- Scikit-learn (SVM)
- NumPy

---

## Usage
1. Clone this repository.
2. Place your dataset in a folder named `Face` (or update path in code).
3. Install dependencies:
```bash
pip install opencv-python numpy scikit-learn tensorflow
4. Run the script
python face_recognition_custom.py
5. Webcam window will open. Press 'q' to exit.
