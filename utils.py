from tensorflow.keras.preprocessing.image import ImageDataGenerator ,load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import random
import numpy as np


def load_data():
    # Paths
    base_dir = "/data"
    train_dir = f"{base_dir}/train_augmented"
    val_dir = f"{base_dir}/val"

    # Rescale generator (no other augmentation here)
    datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    # Load validation data
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    print("Class names:", train_generator.class_indices.key())

    return train_generator, val_generator

def display_data_distribuation(path):
   
    # Count images per class
    train_path = "/content/drive/MyDrive/Datasets/COVID-19_Radiography_Database(splited+no_masks)/"+path
    train_counts = Counter([
        label for folder in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, folder))
        for label in [folder] * len(os.listdir(os.path.join(train_path, folder)))
    ])

    # Create DataFrame
    df = pd.DataFrame(train_counts.items(), columns=['Class', 'Count'])
    df = df.sort_values(by='Count', ascending=False)

    # Plot using Seaborn
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x='Class', y='Count', palette='viridis')
    plt.title('Image Count per Class – Train Set')
    plt.tight_layout()
    plt.show()


def visualize_samples(data_generator):
    # Get one batch of images and labels
    images, labels = next(data_generator)

    # Select 6 random indices from the batch
    indices = np.random.choice(range(images.shape[0]), size=6, replace=False)

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        # Images are normalized (rescaled), so multiply by 255 for display
        img = images[idx]
        plt.imshow(img)
        class_idx = np.argmax(labels[idx])
        class_name = list(data_generator.class_indices.keys())[class_idx]
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



def load_test_data():
    # Paths
    base_dir = "/data"
    test_dir = f"{base_dir}/test"

    # Rescale generator (no other augmentation here)
    datagen = ImageDataGenerator(rescale=1./255)

    # Load test data
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print("Class names:", test_generator.class_indices.keys())

    return test_generator


def visualize_test_predictons(model):
    # Paths
    base_dir = "/data"
    test_dir = f"{base_dir}/test"

    # Class labels
    class_labels = sorted(os.listdir(test_dir))

    # Get 6 random images
    random_images = []
    for _ in range(6):
        cls_dir = random.choice(class_labels)  # pick a random class
        img_name = random.choice(os.listdir(os.path.join(test_dir, cls_dir)))
        random_images.append((os.path.join(test_dir, cls_dir, img_name), cls_dir))

    # Plot
    plt.figure(figsize=(15, 8))
    for i, (img_path, true_label) in enumerate(random_images):
        # Load and preprocess
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred_prob = model.predict(img_array)
        pred_class = class_labels[np.argmax(pred_prob)]

        # Plot
        plt.subplot(2, 3, i + 1)
        plt.imshow(load_img(img_path))
        plt.axis("off")
        plt.title(f"True: {true_label}\nPred: {pred_class}",
                color=("green" if true_label == pred_class else "red"))
    plt.tight_layout()
    plt.show()


def visualize_accuracy_loss():

    # Load the saved history from file
    history = pd.read_csv("training_log.csv")

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("/output/Training_History.png", dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to output folder") 

    plt.show()

def classification_report(model, test_generator):
    
    # Get true labels and predictions
    Y_true = test_generator.classes
    Y_pred_probs = model.predict(test_generator)
    Y_pred = np.argmax(Y_pred_probs, axis=1)

    # Get class labels
    class_labels = list(test_generator.class_indices.keys())

    # Generate classification report
    report = classification_report(Y_true, Y_pred, target_names=class_labels)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    plt.savefig("/output/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to output folder") 

    plt.show()


   
