from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

model = load_model("autoencoder_lstm.h5",compile = False)

scaler = joblib.load("scaler.save")

df = pd.read_csv("synthetic_data.csv")
values = df["value"].values.reshape(-1,1)

values_scaled = scaler.transform(values)

def create_sliding_window(data,size):
    X = []
    for i in range(len(data) - size  + 1):
        X.append(data[i:i +size])
    return np.array(X)

size = 10
X_all = create_sliding_window(values_scaled,size)
print(X_all.shape)
X_all = X_all.reshape((X_all.shape[0],X_all.shape[1],1))

x_pred = model.predict(X_all)

mse_list = np.mean(np.square(X_all-x_pred),axis = (1,2))

threshold = np.percentile(mse_list,95)
print("threshold value : ",threshold)

anomalies = mse_list>threshold

df_result = df.iloc[size - 1:].copy()
df_result["reconstruction_error"] = mse_list
df_result["anomaly"] = anomalies

df_result.to_csv("anomaly_result.csv",index = False)


plt.figure()
plt.plot(df_result["timestamp"],df_result["value"],label = "Data")
plt.scatter(df_result[df_result["anomaly"]]["timestamp"],df_result[df_result["anomaly"]]["value"],c = "red",s = 20,label ="anomaly")
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()