# Kaggle-Digit-Recognizer
# 🧠 Kaggle Digit Recognizer (CNN with TensorFlow)

This project is an end-to-end **computer vision classification pipeline** built for Kaggle’s **Digit Recognizer** competition.  
The goal is to correctly classify handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the MNIST-style dataset.

This project demonstrates my ability to:
- Build and train deep learning models using **TensorFlow / Keras**
- Perform image preprocessing and augmentation
- Design CNN architectures for multi-class classification
- Evaluate model performance and generate predictions for submission

---

## 📌 Project Overview

- **Problem Type:** Multi-class image classification
- **Dataset:** Kaggle Digit Recognizer (28×28 grayscale handwritten digits)
- **Model Type:** Convolutional Neural Network (CNN)
- **Frameworks:** TensorFlow 2.x, Keras
- **Language:** Python

---

## 📂 Dataset Description

- **Training Data (`train.csv`)**
  - 42,000 labeled images
  - Each row represents a 28×28 grayscale image flattened into 784 pixel values
  - Labels range from **0–9**

- **Test Data (`test.csv`)**
  - 28,000 unlabeled images
  - Used to generate Kaggle submission predictions

---

## 🏗️ Model Architecture

### Key Design Choices
- **Batch Normalization** for training stability
- **Data Augmentation** to improve generalization
- **Global Average Pooling** to reduce overfitting
- **Dropout** for regularization
- **Adam optimizer** with categorical cross-entropy loss

---

## ⚙️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Jupyter Notebook**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Chris-Koelsch/Kaggle-Digit-Recognizer.git
cd Kaggle-Digit-Recognizer

The model is a **deep CNN** designed to extract hierarchical spatial features:

