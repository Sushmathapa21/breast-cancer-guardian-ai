# Breast Cancer Guardian AI

An end-to-end deep learning application that detects Invasive Ductal Carcinoma (IDC) in H&E stained histopathology images.

## Overview
This project leverages a fine-tuned MobileNetV2 Convolutional Neural Network (CNN) to classify microscopic tissue cells as either Benign or Malignant. The model was trained on a highly imbalanced dataset of 277,524 images, utilizing class weights to prioritize the detection of malignant cases and minimize false negatives.

## Tech Stack
* **Deep Learning Framework:** TensorFlow & Keras (MobileNetV2)
* **Data Processing:** NumPy, Pillow, Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend UI:** Streamlit

## Model Performance
* **Overall Accuracy:** ~85%
* **Recall (Malignant):** ~83% (Successfully optimized to catch malignant patterns)

## How to Run Locally
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/breast-cancer-guardian-ai.git`
2. Install the requirements: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`

## Disclaimer
This is an AI tool designed for educational and portfolio purposes. It is not intended for medical or clinical diagnosis.