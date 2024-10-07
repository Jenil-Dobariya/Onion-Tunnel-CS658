import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from PIL import Image
import os


data = pd.read_csv('VPN120s.csv')


data.iloc[:, -1] = data.iloc[:, -1].replace({'Non-VPN':'non-darknet', 'VPN':'vpn'})
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()
train_data = data
print(data)

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)

importances = model.feature_importances_

feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

important_features = feature_ranking[feature_ranking['Importance'] > 0.001]

selected_features = important_features.head(61)

# print("Selected Features:")
# print(selected_features)

labels = selected_features.iloc[:, 0].tolist()

data_labels = labels
data_labels.append('class1')

filtered_data = data[data_labels]

# print(filtered_data)

filtered_data.to_csv('filtered_vpn_data.csv', index=False)
filtered_data = pd.read_csv('filtered_vpn_data.csv')

# print(filtered_data)

filtered_tor_data = filtered_data.iloc[:, :-1]
filtered_tor_lbl = filtered_data.iloc[:, -1]

# print(filtered_tor_lbl)


output_dir = input("Enter name of directory to specify iamge storage location: ")
os.makedirs(output_dir, exist_ok=True)

image_info_df = pd.DataFrame(columns=['image_path', 'Label'])

filtered_darknet_df = (filtered_tor_data - filtered_tor_data.min()) / (filtered_tor_data.max() - filtered_tor_data.min())

for index, row in filtered_darknet_df.iterrows():
    # Convert the row to a numpy array
    image_data = row.values

    # Pad the array with zeros to make it 64 values
    if len(image_data) < 64:
        padded_data = np.pad(image_data, (0, 64 - len(image_data)), 'constant')
    else:
        padded_data = image_data[:64]  # Trim if more than 64 values

    # Reshape to (8, 8)
    image_data_reshaped = padded_data.reshape(8, 8)

    # Normalize the data to range [0, 255] for grayscale
    image_data_normalized = (image_data_reshaped * 255).astype(np.uint8)

    # Create a grayscale image from the array
    image = Image.fromarray(image_data_normalized, mode='L')  # 'L' mode for grayscale

    # Save the image
    image_path = f"{output_dir}/image_{index}.png"
    image.save(image_path)

    # Append the image path and last two columns directly into the DataFrame
    image_info_df = image_info_df._append({
        'image_path': image_path,
        'Label': filtered_tor_lbl.iloc[index],
    }, ignore_index=True)

    print(f"{image_path} is done")

image_info_df.to_csv('image_info_vpn.csv', index=False)

