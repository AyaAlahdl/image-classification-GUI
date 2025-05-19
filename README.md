# Image Classification with 6 Categories & Stacked Ensemble Meta-Model + GUI

This project applies advanced image classification techniques using multiple deep learning models combined through a **stacked meta-model ensemble** approach. The system classifies images into six predefined categories and offers a user-friendly GUI for real-time interaction and prediction.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Models and Techniques](#models-and-techniques)  
- [Key Features](#key-features)  
- [Installation](#installation)  

---

## Project Overview

The goal of this project is to build a reliable image classification pipeline that leverages multiple pretrained and custom deep learning models. By stacking these base models and feeding their prediction outputs into a meta-classifier, the ensemble achieves higher accuracy and robustness compared to any single model alone. A graphical user interface enables users to easily load images and models, then view predictions with confidence scores.

---

## Models and Techniques

- **Custom CNN:** Specifically designed and trained for the classification task.  
- **VGG16 & VGG19:** Pretrained convolutional networks fine-tuned for feature extraction.  
- **ResNet:** Deep residual network to improve training of very deep architectures.  
- **InceptionV3:** Captures multi-scale features with varying convolution sizes.  
- **DenseNet:** Dense connectivity improves feature reuse and reduces parameters.  
- **Stacked Meta-Model Ensemble:** Softmax outputs from base models are combined as input features to a Logistic Regression meta-classifier, enhancing final prediction accuracy.

---

## Key Features

- Image preprocessing including resizing, normalization, and optional augmentation.  
- Ensemble learning combining multiple models’ outputs for improved accuracy.  
- Detailed evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix).  
- Interactive GUI for uploading images, loading models, and displaying predictions with confidence.  
- Support for loading base models and stacked meta-model dynamically.

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/Image-Classification-Project--Six-Categories-with-GUI.git
    cd Image-Classification-Project--Six-Categories-with-GUI
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

> **Note:** This project requires Python 3.8 and packages including TensorFlow, OpenCV, scikit-learn, PyQt5 (or Tkinter), NumPy, and Matplotlib.

---

## Usage

Run the application:

```bash
python GUI app.py
