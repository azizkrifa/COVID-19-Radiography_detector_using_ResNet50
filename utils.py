from tensorflow.keras.preprocessing.image import ImageDataGenerator ,load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
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


def visualize_accuracy_loss(history):
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    #Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("/output", dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to output") #save to output folder

    plt.show()

def classification_report(model, test_generator):
    
    # Get true labels and predictions
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
    
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred_classes)
    
    print("Confusion Matrix:")
    print(cm)

   
