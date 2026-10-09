"""
Definition of Project : Anomaly detection with Lstm Autoencoder

dataset: a synthetic dataset:
            - sinus wave and noise
            

libs: tensorflow,scikit-learn,pandas, numpy, seaborn,matplotlib

plan:
    - load and explore the data
    - preprocess
    - model
    - test


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset

np.random.seed(42)
t = np.arange(0,1000)
data = np.sin(0.02*t) + 0.1 * np.random.normal(size = len(t))

data[200:210] += 2 # positive anomaly
data[500:510] -= 3 # negative anomaly
data[800:805] += 3.5

df = pd.DataFrame({
    "timestamp":pd.date_range(start = "2026-01-01",periods = len(t),freq = "H"),
    "value": data
})

print(df.head())

df.to_csv("synthetic_data.csv",index = False)


plt.figure()
plt.plot(df["timestamp"],df["value"],label ="Values")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid("on")
plt.tight_layout()
plt.show()