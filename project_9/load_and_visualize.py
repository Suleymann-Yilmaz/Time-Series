"""
Problem Definition : Early Earthquake Detection System with 1D CNN

Dataset : Synthetic Dataset

Class: 1 - earthquake
       0 - not eartquake

lib: tensorflow/keras, scikit-learn,numpy,matplotlib,seaborn

Plan:
1- Creating Sythetic dataset
2- Preprocessing
3- Model and train
4- test the model

"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.makedirs("time-series/data",exist_ok=True)

# parameters
sample = 1500
num_signal = 1000 

def generate_eart():
    t = np.linspace(0,1,sample)
    freq = np.random.uniform(5,15)
    envelope = np.exp(-((t-0.5)**2)/0.01)
    signal = envelope*np.sin(2*np.pi*freq*t)
    noise = np.random.normal(0,1,sample)
    return signal + noise


dummy = generate_eart()

def generate_noise():
    t = np.linspace(0,1,sample)
    base = np.sin(2*np.pi*np.random.uniform(0.1,1,1)*t)
    noise = np.random.normal(0,0.3,sample)
    return base + noise

dummy = generate_noise()


X,y = [],[]

for _ in range(num_signal//2):
    X.append(generate_eart())
    y.append(1)


for _ in range(num_signal//2):
    X.append(generate_noise())
    y.append(0)


X = np.array(X)
y = np.array(y)

indicies = np.arange(len(X))
np.random.shuffle(indicies)

X = X[indicies]
y = y[indicies]

np.save("data/X_signal.npy",X)
np.save("data/y_label.npy",y)

plt.figure()
for i in range(3):
    plt.plot(X[i],label = f"{y[i]}")

plt.title("Sample Signal")
plt.legend()
plt.show()