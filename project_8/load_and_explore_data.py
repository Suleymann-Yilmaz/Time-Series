"""
Definition  : Sensor Fault Detection with LSTM-FCN

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("secom.data",sep =r"\s+",header = None)
print(df.head())

labels = pd.read_csv("secom_labels.data",sep = r"\s+",header = None)
print(labels.head())

df["label"] = labels[0].values

print(df.head())

print(df.shape)
print(df["label"].value_counts(normalize=True))


# missing value

missing_value = df.isnull().sum()
total_missing_value = missing_value.sum()
print(f"Total Missing value : ",total_missing_value)


missing_ratio = (missing_value/len(df))*100
missing_ratio = missing_ratio[missing_ratio>0].sort_values(ascending=False)
print(missing_ratio)


plt.figure()
sns.barplot(x = missing_ratio.index,y =missing_ratio.values)
plt.title("Missing ratio")
plt.xlabel("Index")
plt.ylabel("Missing value counts")
plt.xticks([])
plt.show()


plt.figure()
sns.scatterplot(data =df,x = 0,y = 1,hue = "label",alpha = 0.6,palette="Set1")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.tight_layout()
plt.grid()
plt.show()

df.to_csv("data-secom.csv",index = False)