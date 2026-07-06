import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained CNN
cnn_model = load_model("cnn_wound_model.h5")

# 🔥 FORCE BUILD by calling model once
dummy_input = tf.zeros((1, 224, 224, 3))
cnn_model(dummy_input)

# Create feature extractor model
feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].output
)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    features = feature_model.predict(img)
    return features.flatten()

if __name__ == "__main__":
    test_image = "Data/cnn_raw_images/IMG435.jpg"  # put ANY wound image path here
    features = extract_features(test_image)
    print("Feature vector shape:", features.shape)

