import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

IMG_SIZE = (8, 8)

model = load_model("trained_model.h5")
print("Model loaded successfully.")

model.summary()

csv_file = input("enter which csv you want to test ( containing image info ): ")
data = pd.read_csv(csv_file)

image_paths = data.iloc[:, 0].values     # First column: image paths
labels = data.iloc[:, -1].values         # Second last column: 4-class labels

# print(image_paths)

# Load and preprocess images
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")  # Load image and resize
    image = img_to_array(image)  # Convert to numpy array
    image = image / 255.0        # Normalize pixel values to [0, 1]
    return image

# Preprocess all images in the test dataset
images = np.array([load_and_preprocess_image(img_path) for img_path in image_paths])

# Encode labels to integer format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# # print(y_val)

# y_pred = model.predict(X_val)
# y_pred_classes = np.argmax(y_pred, axis=1)  

# print("\nClassification Report:")
# print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

# Make predictions on the test dataset
# test_loss, test_acc = model.evaluate(images, labels_encoded)

y_pred = model.predict(images)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class indices

# Print classification report
print("\nClassification Report:")
print(classification_report(labels_encoded, y_pred_classes, target_names=['non-darknet', 'tor', 'vpn']))
