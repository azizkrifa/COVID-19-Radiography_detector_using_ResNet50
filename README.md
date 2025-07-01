# 🦠 COVID-19-Radiography_detector_using_ResNet50

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

- ✅ Data `preprocessing` and `augmentation`
- ✅ Train/validation/test split using `splitfolders`
- ✅ CNN model built on top of pre-trained architectures (`ResNet50`)
- ✅ `EarlyStopping` and `ModelCheckpoint` callbacks for efficient training.
- ✅ Visualization of training / validation `accuracy` and `loss`.
- ✅ Final model export (`.h5`) and `training history` logging.
- ✅ Test set evaluation and `classification report` / `confusion matrix` visualization.

---

## 🗂 Dataset

- **COVID-19 Radiography Database**
  - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
  - Classes: `COVID-19`, `Normal`, `Viral Pneumonia`, `Lung Opacity`.
  - split into `train(70%)`, `val(20%)`, and `test(10%)` folders using **split-folders** library.
  - Preprocessed with resizing, normalization, and augmentation.
 
### 📊 Class Distribution :

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b098fe2-0684-4f94-9827-639f151b5f57" width="47%" height="300px" />
  <img src="https://github.com/user-attachments/assets/52d0ebca-9886-4c65-afec-2dd9bc56a022" width="47%" height="300px" />
</p>




📝 **Note**:
The original COVID-19 Radiography Database contains over `41,000 files`, including approximately `21,000 chest X-ray images` and an equal number of corresponding `segmentation masks`. However, since this project focuses solely on `image classification` and not segmentation or object detection, only the X-ray images are used in the dataset. The segmentation masks are excluded from the training pipeline.

👉 You can access the **post-processed** dataset (ready to fit the model) used in this project here: [post-processed dataset](https://drive.google.com/drive/folders/1PAAzK57hLqxDldSSguqaG2s80R4yoHUr?usp=sharing)

---

## 🧪 Model Setup

- **Base model**: `ResNet50` (pretrained on ImageNet, `include_top=False`).
  
- **Input size**: 224×224×3 (**RGB**).
  
- **Architecture**:
  
  - GlobalAveragePooling2D
  - **Dense**(512, **ReLU**) + **BatchNormalization**.
  - **Dense**(256, **ReLU**) + **BatchNormalization**.
  - **Dense**(10, **Softmax**). `output layer`
    
---

## 📊 Training Strategy

- Optimizer: `Adam`(learning rate = 1e-5).
- Loss: `Categorical Crossentropy`.
- Metrics: `Accuracy`.
- Epochs: Up to 50 with EarlyStopping (`patience=10`).
- Callbacks:
  - `ModelCheckpoint`: saves best model (`best_model.h5`)
  - `EarlyStopping`: avoids overfitting and saves time

---

## 🚀 How to Run

### 📥 1. Clone the Repository
``` bash 
    git clone https://github.com/azizkrifa/COVID-19-Radiography_detector_using_ResNet50.git
    cd  COVID-19-Radiography_detector_using_ResNet50
```

### 📦 2. Install Dependencies

``` bash 
   pip install -r requirements.txt
``` 
### 🧪 3. Run the Pipeline Step-by-Step

   ####  🧹 3.1: Data Preparation

  Load and preprocess the dataset (including splitting and augmentation):

  📄 Run: `data_preprocessing.ipynb`

  ####  🧠 3.2: Train the Model

  Train the model on the prepared dataset: 
  
  📄 Run: `training.ipynb`
    
  #### 📊 3.3: Evaluate the Model

  Evaluate model performance and visualize results: 
  
  📄 Run: `evaluation.ipynb`
     
---

## 📁 Model Outputs & Evaluation Results

The following metrics and visualizations were generated to assess the model’s performance:

- **Training History**: [📁 See `output/training_history.png`]  
  Shows model accuracy and loss across epochs.

- **Classification Report**: [📁 See `output/classification_report.png`]  
  High performance across all classes, with a test accuracy of **94%**.

- **Confusion Matrix**: [📁 See `output/confusion_matrix.png`]  
  Visual breakdown of model predictions vs. actual labels.

- **Trained Model File**: [📁 See `output/best_model.h5`]  
  The best-performing model saved during training using `ModelCheckpoint`.

➡️  All evaluation outputs and the final model are stored in the `output/` folder.

---

## 🔬 Sample Predictions 
![Some predictions ](https://github.com/user-attachments/assets/adfbf6e4-c1d0-4e93-8482-0fd2d50f14c9)





