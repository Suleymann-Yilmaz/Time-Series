import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

data_dir = "data"

X = np.load(os.path.join(data_dir,"X_signal.npy"))
y = np.load(os.path.join(data_dir,"y_label.npy"))


def z_score_normalize(signal_batch):
    normalized = []
    for signal in signal_batch:
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            std = 1
        norm_signal = (signal - mean)/std
        normalized.append(norm_signal)
    return np.array(normalized)

X_normalized = z_score_normalize(X)

X_train,X_test,y_train,y_test = train_test_split(X_normalized,y,test_size=0.2,random_state=42,stratify=y)

print("X_train : ",X_train.shape)
print("X_test : ",X_test.shape)
print("y_train : ",y_train.shape)
print("y_test : ",y_test.shape)


np.save("data/X_train.npy",X_train)
np.save("data/X_test.npy",X_test)
np.save("data/y_train.npy",y_train)
np.save("data/y_test.npy",y_test)

print("They are saved")

