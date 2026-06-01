# Kaggle Digit Recognizer (CNN Model Using TensorFlow Keras Library)

This project is an end-to-end computer vision number classification pipeline.  
The goal is to correctly classify handwritten digits (0–9) using a CNN trained on the MNIST-style dataset.

This project demonstrates my ability to:
- Build and train deep learning models using TensorFlow / Keras
- Perform image preprocessing and augmentation
- Design CNN architectures for multi-class classification
- Evaluate model performance and generate predictions for submission

---

## Overview

- **Problem Type:** Multi-class image classification
- **Dataset:** Kaggle Digit Recognizer (28×28 grayscale handwritten digits)
- **Model Type:** Convolutional Neural Network (CNN)
- **Frameworks:** TensorFlow 2.x, Keras
- **Language:** Python

---

## Dataset Description

- **Training Data (`train.csv`)**
  - 42,000 labeled images
  - Each row represents a 28×28 grayscale image flattened into 784 pixel values
  - Labels range from **0–9**

- **Test Data (`test.csv`)**
  - 28,000 unlabeled images
  - Used to generate Kaggle submission predictions

---

## Model Architecture

### Design Choices
- **Batch Normalization** for training stability
- **Data Augmentation** to improve generalization
- **Global Average Pooling** to reduce overfitting
- **Dropout** for regularization
- **Adam optimizer** with categorical cross-entropy loss

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Jupyter Notebook**

---

## How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Chris-Koelsch/Kaggle-Digit-Recognizer.git
cd Kaggle-Digit-Recognizer


