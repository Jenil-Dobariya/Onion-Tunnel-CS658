import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv('Darknet.csv')
data = data.dropna()


last_column = data.columns[-1]
second_last_column = data.columns[-2]
data[last_column] = data[last_column].str.lower()
data[second_last_column] = data[second_last_column].str.lower()
data.iloc[:, -2] = data.iloc[:, -2].replace({'non-tor':'non-darknet', 'nonvpn':'non-darknet'})
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()
train_data = data[~data.iloc[:, -1].isin(['p2p', 'browsing'])]

X = pd.concat([train_data.iloc[:, 5], train_data.iloc[:, 7:-2]], axis=1)
y = train_data.iloc[:, -1]

model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)

importances = model.feature_importances_

feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

important_features = feature_ranking[feature_ranking['Importance'] > 0.001]

selected_features = important_features.head(64)

selected_features.to_csv("selected_features.csv")

selected_features['Feature'] = selected_features['Feature'].apply(lambda x: x.replace('Packet', 'Pkt')
                                               .replace('Segment', 'Seg')
                                               .replace('Total', 'Tot')
                                               .replace('Length', 'Len')
                                               .replace('Count', 'Cnt')
                                               .replace('Variance', 'Var')
                                               .replace('Bulk', 'Blk')
                                               .replace('Byte', 'Byt'))

selected_features['Feature'] = selected_features['Feature'].replace('FWD Init Win Byts', 'Init Fwd Win Byts')
selected_features['Feature'] = selected_features['Feature'].replace('Bwd Init Win Byts', 'Init Bwd Win Byts')
selected_features['Feature'] = selected_features['Feature'].replace('Average Pkt Size', 'Pkt Size Avg')
selected_features['Feature'] = selected_features['Feature'].replace('Tot Len of Fwd Pkt', 'TotLen Fwd Pkts')
selected_features['Feature'] = selected_features['Feature'].replace('Tot Len of Bwd Pkt', 'TotLen Bwd Pkts')
selected_features['Feature'] = selected_features['Feature'].replace('Tot Bwd packets', 'Tot Bwd Pkts')
selected_features['Feature'] = selected_features['Feature'].replace('Tot Fwd Pkt', 'Tot Fwd Pkts')
selected_features['Feature'] = selected_features['Feature'].replace('Bwd Pkt/Blk Avg', 'Bwd Pkts/b Avg')

selected_features.to_csv("selected_features_pcap.csv")