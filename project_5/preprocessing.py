import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

df = pd.read_csv("synthetic_data.csv")
print(df.head())

values = df["value"].values.reshape(-1,1)
print(values)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
print(scaled)

joblib.dump(scaler,"scaler.save")


def create_sliding_window(data,window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])

    return np.array(X)

window_size = 10
X = create_sliding_window(scaled,window_size)

X = X.reshape((X.shape[0],X.shape[1],1))
np.save("X_train.npy",X)