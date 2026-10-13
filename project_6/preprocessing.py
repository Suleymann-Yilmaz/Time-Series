import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("air_quality_cleaned.csv")
print(data.head())
data.set_index("Datetime",inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler,"scaler_air.save")


def create_window_sliding(data,window_size):
    X,y = [],[]
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i+window_size][0])

    return np.array(X),np.array(y)

window_size = 72
X,y = create_window_sliding(scaled_data,window_size = window_size)


print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

print(f"X_train shape",X_train.shape)
print(f"X_test shape",X_test.shape)
print(f"y_train shape",y_train.shape)
print(f"y_test shape",y_test.shape)


np.save("X_train_ls.npy",X_train)
np.save("X_test_ls.npy",X_test)
np.save("y_train_ls.npy",y_train)
np.save("y_test_ls.npy",y_test)