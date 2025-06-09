# 🤟 ASL Alphabet Classification Using Deep Learning

A deep learning-based system to classify static American Sign Language (ASL) hand gestures with high accuracy and real-time inference capabilities. Built using TensorFlow, EfficientNetB0, and TensorFlow Lite for deployment.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing & Augmentation](#preprocessing--augmentation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Real-Time Inference](#real-time-inference)
- [Active Learning](#active-learning)
- [Deployment](#deployment)
- [Challenges & Solutions](#challenges--solutions)
- [Applications](#applications)
- [Future Work](#future-work)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [License](#license)

---

## 📌 Overview

This project focuses on recognizing **29 distinct static ASL gestures**—26 alphabet letters and 3 special classes: `'del'`, `'nothing'`, and `'space'`. It leverages **transfer learning** with **EfficientNetB0**, data augmentation, hyperparameter tuning, and real-time webcam inference using **TFLite**.

Key goals:
- Achieve >90% accuracy in validation.
- Ensure real-time inference performance.
- Develop a robust preprocessing and augmentation pipeline.
- Explore active learning for improving performance on ambiguous classes.

---

## 📊 Dataset

- **Source:** ASL Alphabet Dataset (static images).
- **Classes:** 29 (A–Z + `del`, `nothing`, `space`).
- **Image Count:** 
  - 11,600 training images (500/class)
  - 2,900 validation images (20% split)
  - 2,900 test images
- **Format:** Images stored in class-labeled directories.
- **Distribution:** Uniform across all classes.

---

## 🧼 Preprocessing & Augmentation

### Preprocessing Pipeline

| Stage            | Description                                              |
|------------------|----------------------------------------------------------|
| Resizing         | All images resized to 128x128x3                          |
| Normalization    | `EfficientNetB0.preprocess_input()` → range `[-1, 1]`    |
| Validation Split | 80/20 split within training set                          |
| Test Set Issue   | Incorrect scaling (`[0, 1]`) caused performance drop     |

> ⚠️ **Test Preprocessing Bug**: Mismatched scaling caused a 3.45% test accuracy, later corrected.

### Augmentation Techniques

- Rotation ±10°
- Width/Height Shift ±10%
- Shear ±10%
- Zoom ±10%
- Horizontal Flip
- Fill Mode: `'nearest'`

> Augmentation is applied **only to training data** to avoid leakage.

---

## 🏗️ Model Architecture

### 🔸 Custom CNN (Baseline)
- 3 Conv2D → BatchNorm → MaxPooling layers
- Dense (256) + Dropout (0.5) → Dense (29, Softmax)
- ~6.5M parameters

### 🔹 EfficientNetB0 (Transfer Learning)
- Pretrained on ImageNet (top removed)
- GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(29, Softmax)
- Trainable Parameters: ~1.7M

### 🔧 Hyperparameter Optimization
- Optimizer: Adam
- Learning Rates:
  - Initial: `1e-4`
  - Fine-tune: `1e-5`
  - Tuned: `5e-4`
- Dropout: 0.3 (optimal via Keras Tuner)
- Batch Size: 32
- Loss: Categorical Crossentropy

---

## 🏋️ Training Process

| Phase        | Highlights                                                  |
|--------------|-------------------------------------------------------------|
| Initial      | Achieved 91.1% validation accuracy in 17 epochs             |
| Fine-Tuning  | Unfroze top 20 EfficientNetB0 layers                        |
| Retraining   | Added 100 samples for 'X' and 'U'; hit 93.58% val accuracy  |
| Callbacks    | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau           |

> Training time: ~528 seconds per epoch on NVIDIA RTX 3050 GPU

---

## ✅ Evaluation

| Metric                  | Result                |
|-------------------------|-----------------------|
| Training Accuracy       | 98.9%                 |
| Validation Accuracy     | 93.6% (after retraining) |
| Test Accuracy           | 3.45% → **Fixed** to align preprocessing |
| Avg F1-Score (Validation)| ~90%                |
| Inference Latency       | ~80ms/frame (real-time capable) |

### Confusion Matrix Observations
- High accuracy across most classes.
- Frequent confusions:
  - `'M'` vs `'N'`
  - `'R'` vs `'S'`
  - `'X'` vs `'U'`
  - `'nothing'` vs `'space'`

### Solutions:
- Increase input resolution
- Add attention mechanisms (e.g., CBAM)
- Use depth or multi-view images

---

## 🎥 Real-Time Inference

- Deployed using **TensorFlow Lite** + OpenCV
- Real-time webcam inference (~60–100ms/frame)
- Issues:
  - Prediction stuck on one class (`'B'`)
  - No localization of hand region
- Proposed Enhancements:
  - Add **MediaPipe** or **YOLO** for hand detection
  - Apply **temporal smoothing** (e.g., moving average of predictions)

---

## 🔁 Active Learning

- Additional samples for misclassified letters (`X`, `U`) collected via webcam.
- Data pipeline supports live data collection and retraining.
- Result: Validation accuracy increased to **93.58%** in just 1 epoch.

---

## 🚀 Deployment

- Converted model to `.tflite` for mobile/embedded devices.
- Model size and latency optimized for edge inference.

---

## 🧩 Challenges & Solutions

| Challenge                          | Solution                                                   |
|-----------------------------------|------------------------------------------------------------|
| Test accuracy anomaly             | Align preprocessing with training pipeline                 |
| Misclassification of similar signs| Collect more data, use attention and localization          |
| Real-time instability             | Integrate hand tracking, background filtering              |
| Overfitting risk                  | Data augmentation + dropout tuning                         |

---

## 💡 Applications

- 🔊 **Accessibility:** Enable real-time communication for Deaf/HoH individuals.
- 🎓 **Education:** Interactive tools for ASL learners.
- 💻 **HCI:** Gesture-based control interfaces for apps and games.

---

## 🔮 Future Work

- Integrate dynamic gesture recognition (`J`, `Z`) via RNNs or 3D CNNs.
- Expand dataset with varied lighting, backgrounds, and skin tones.
- Apply hand segmentation/localization (MediaPipe, YOLO).
- Deploy on mobile devices with hardware acceleration.
- Ensemble models to enhance classification accuracy.

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/asl-alphabet-classification.git
cd asl-alphabet-classification

# Install dependencies
pip install -r requirements.txt

```
--- 

```
📁 asl-alphabet-classification/
├── model/
│ ├── asl_best_model.keras # Best Keras model saved
│ └── asl_model.tflite # Exported TensorFlow Lite model
├── tuner_dir/
│ └── asl_tuner/ # Keras Tuner trials and metadata
│ ├── trial_0000/ to trial_0029/
│ ├── oracle.json
│ └── tuner0.json
├── ASL Alphabet.ipynb # Main training & evaluation notebook
├── LICENSE # MIT License file
├── README.md # Project documentation (this file)
├── Report.pdf # Comprehensive project report

```

---

**Dataset Source:**  
Due to size constraints, the dataset is not included in this repository. You can download it from Kaggle:  
📦 [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## ⚖️ License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full details.


---

## 🔗 Contact & Links

- **GitHub Profile:** [parmod2310](https://github.com/parmod2310)
- **Email:** [p921035@gmail.com](mailto:p921035@gmail.com)

Feel free to reach out for collaboration, feedback, or contributions!

---

## 💬 Visitor Note

> 👋 **Hello and welcome!**

Thank you for visiting the ASL Alphabet Classification project repository.

This system represents a step toward more inclusive communication using AI.  
If you have questions, suggestions, or would like to contribute, feel free to open an issue or reach out via email.

Enjoy exploring the repo and keep building amazing things! 🚀

---
