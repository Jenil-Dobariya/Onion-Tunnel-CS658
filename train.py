import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, Flatten, Dense
# from tf.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys

IMG_SIZE = (8, 8)  
BATCH_SIZE = 32    
EPOCHS = 100       


def encode_labels(arr):
    mapping = {'tor': 1, 'vpn': 2, 'non-darknet': 0}
    return np.array([mapping.get(item, -1) for item in arr])

def decode_labels(arr):
    reverse_mapping = {1: 'tor', 2: 'vpn', 0: 'non-darknet'}
    return np.array([reverse_mapping.get(item, 'unknown') for item in arr])

filename = sys.argv[1]

data = pd.read_csv(filename)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()

data['Label'] = data['Label'].str.lower()
data['Label'] = data['Label'].replace({'non-tor':'non-darknet', 'nonvpn':'non-darknet'})

features_df = pd.read_csv('selected_features_pcap.csv')
if filename == 'Darknet.csv':
    features_df = pd.read_csv('selected_features.csv')
features = features_df["Feature"].tolist()

X = data[features]
y = data['Label']

X_normalized = (X - X.min()) / (X.max() - X.min())

images = []

for index, row in X_normalized.iterrows():
    image_data = row.values

    if len(image_data) < 64:
        padded_data = np.pad(image_data, (0, 64 - len(image_data)), 'constant')
    else:
        padded_data = image_data[:64]

    image_data_reshaped = padded_data.reshape(8, 8)

    image_data_normalized = (image_data_reshaped * 255).astype(np.uint8)

    image = Image.fromarray(image_data_normalized, mode='L')
    image_array = img_to_array(image) / 255.0

    images.append(image_array)

images = np.array(images)
labels = y.values
labels_encoded = encode_labels(labels)

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
y_pred_classes = decode_labels(y_pred_classes)
y_val = decode_labels(y_val)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes))

model.save('model.h5')