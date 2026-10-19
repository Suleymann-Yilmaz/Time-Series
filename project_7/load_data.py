"""
Definition : Detect Human Activity with Inception

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_inertial(folder_path,subset = "train"):
    signal_names = [
        "body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z"
    ]

    signal_data = []
    for signal in signal_names:
        filename = f"{signal}_{subset}.txt"
        filepath = os.path.join(folder_path,filename)
        data = np.loadtxt(filepath)
        signal_data.append(data)

    stacked = np.stack(signal_data,axis = 0).transpose(1,2,0)
    return stacked
# train set
X_train = load_inertial("UCI HAR Dataset/train/Inertial Signals","train")
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype("int") - 1
print(X_train.shape)
print(y_train.shape)

X_test = load_inertial("UCI HAR Dataset/test/Inertial Signals","test")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype("int") - 1

print(X_test.shape)
print(y_test.shape)

np.save("X_train_raw.npy",X_train)
np.save("X_test_raw.npy",X_test)
np.save("y_test_raw.npy",y_test)
np.save("y_train_raw.npy",y_train)