# 🦠 COVID-19 Radiography Detector

A deep learning project for the **automated detection of COVID-19** from chest X-ray images using Convolutional Neural Networks (CNNs).

---

## 🧠 Overview

This project leverages transfer learning and CNN-based architectures to classify chest X-ray images into:

- **COVID-19**
- **Normal**
- **Viral Pneumonia**
- **Lung Opacity**

The goal is to assist radiologists and healthcare professionals in making quicker and more accurate diagnoses using AI-based support tools.

---

## ⚙️ Features

- ✅ Data preprocessing and augmentation
- ✅ Train/validation/test split using `splitfolders`
- ✅ CNN model built on top of pre-trained architectures (`ResNet50`)
- ✅ EarlyStopping and ModelCheckpoint callbacks for efficient training
- ✅ Visualization of training/validation accuracy and loss
- ✅ Final model export (`.h5`) and training history logging (`.pkl`)

- ✅ Test set evaluation and confusion matrix visualization

---

## 🗂 Dataset

- **COVID-19 Radiography Database**
  - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
  - Classes: `COVID-19`, `Normal`, `Viral Pneumonia`, `Lung Opacity`
  - Preprocessed with resizing, normalization, and augmentation

---

## 🧪 Model Architecture

- Base Model: `ResNet50` 
- Custom Top Layers:
  - `GlobalAveragePooling2D`
  - Dense layers (e.g., 256 units + ReLU)
  - Final output: `Dense(4, activation='softmax')`

---

## 📊 Training Strategy

- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- Metrics: `Accuracy`
- Epochs: Up to 50 with EarlyStopping (`patience=10`)
- Callbacks:
  - `ModelCheckpoint`: saves best model (`best_model.h5`)
  - `EarlyStopping`: avoids overfitting and saves time

---

## 📊 Outputs

- Final model: `best_model.h5`
- Training history: `history.pkl`
- Accuracy/Loss plot: `training_metrics.png`
- Evaluation metrics on test set (accuracy, confusion matrix, classification report)

---


## 🚀 How to Run

# Clone repo
git clone https://github.com/azizkrifa/COVID-19-Radiography_detector.git
cd COVID19-Radiography-Detector

# Install dependencies
pip install -r requirements.txt

#1-load the data (split + augmentation) by running  1_data_preprocessing.ipynb

#2-Train the model by running  2_training.ipynb

#3-Evaluate the model by running 3_evaluation.ipynb


## 📁 Final Artifacts Location

✅ Check the output folder where you will find the final model + training history (accuracy and loss):

- best_model.h5
- history.pkl
- Classification_Report.png

## Some predictions
![Some predictions ](https://github.com/user-attachments/assets/adfbf6e4-c1d0-4e93-8482-0fd2d50f14c9)



