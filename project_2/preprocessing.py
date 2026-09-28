import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv.gz")

data["date_time"] = pd.to_datetime(data["date_time"])
data.set_index("date_time",inplace = True)

data["hour"] = data.index.hour
data["dayofweek"] =  data.index.dayofweek
data["month"] = data.index.month


features = ["temp","rain_1h","snow_1h","clouds_all","hour","dayofweek","month"] # input_features
target = ["traffic_volume"]

data = data[features + target].dropna()

scaler_x = MinMaxScaler() # features
scaler_y = MinMaxScaler() # target

X_scaled = scaler_x.fit_transform(data[features])
y_scaled = scaler_y.fit_transform(data[target])

joblib.dump(scaler_x,"scaler_x.save")
joblib.dump(scaler_y,"scaler_y.save")

# create windowing
def create_seq(X,y,seq_len):
    X_seq,y_seq = [],[]
    for i in range(len(X)-seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq),np.array(y_seq)

Seq_len = 24
X_seq,y_seq = create_seq(X_scaled,y_scaled,Seq_len)

# split-data

split_index = int(0.8*len(X_seq))
X_train = X_seq[:split_index]
X_test = X_seq[split_index:]
y_train  = y_seq[:split_index]
y_test = y_seq[split_index:]

np.save("X_train.npy",X_train)
np.save("X_test.npy",X_test)
np.save("y_train.npy",y_train)
np.save("y_test.npy",y_test)