import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class CovidModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_names = ["COVID","Lung Opacity","Normal","Viral Pneumonia"]

    def predict(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = self.model.predict(x)
        idx = np.argmax(preds[0])
        label = self.class_names[idx]
        confidence = preds[0][idx]
        return label, confidence
