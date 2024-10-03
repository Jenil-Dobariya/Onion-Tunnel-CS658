import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, Flatten, Dense
# from tf.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Parameters
IMG_SIZE = (8, 8)  # Image size (8x8 pixels)
BATCH_SIZE = 32    # Batch size for training
EPOCHS = 100       # Number of epochs

csv_file = "image_info.csv"  # Path to your CSV file
data = pd.read_csv(csv_file)

image_paths = data.iloc[:, 0].values     # First column: image paths
labels = data.iloc[:, -1].values         # Second last column: 4-class labels

def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")  # Load image and resize
    image = img_to_array(image)  # Convert to numpy array
    image = image / 255.0        # Normalize pixel values to [0, 1]
    return image

images = np.array([load_and_preprocess_image(img_path) for img_path in image_paths])
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 4 classes now
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    verbose=1)

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)  

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

model.save('trained_model.h5')