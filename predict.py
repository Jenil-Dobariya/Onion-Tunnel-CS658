import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import sys


IMG_SIZE = (8, 8)

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

model = load_model("model_combined.h5")
print("Model loaded successfully.")

y_pred = model.predict(images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_classes = decode_labels(y_pred_classes)

usage_type = 0

if 'tor' in y_pred_classes:
    usage_type += 1
if 'vpn' in y_pred_classes:
    usage_type += 2

if usage_type == 3:
    print("ALERT : Usage of Tor and VPN is detected !!!")
elif usage_type == 2:
    print("ALERT : Usage of VPN is detected !!!")
elif usage_type == 1:
    print("ALERT : Usage of Tor is detected !!!")
else:
    print("Any darknet traffic is not detected")
