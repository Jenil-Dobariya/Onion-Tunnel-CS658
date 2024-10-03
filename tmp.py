import pandas as pd
import numpy as np

data = pd.read_csv('Tor.csv')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()
data.iloc[:, -1] = data.iloc[:, -1].replace({'nonTOR':'non-darknet', 'TOR':'tor'})
train_data = data[data.iloc[:, -1].isin(['tor'])]

print(train_data)