"""
Definition of Project-6 : Prediction of Air Quality with LSTM

Plan:
    - Load data and explore it 
    - Preprocessing
    - LSTM Model
    - Apı
    - Streamlit

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import seaborn as sns

df = pd.read_csv("AirQualityUCI.csv",sep = ";",decimal  = ",",encoding = "latin1")
print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())
df.dropna(axis = 1,how = "all",inplace = True)
print(df.isnull().sum())

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],format = "%d/%m/%Y %H.%M.%S",errors="coerce")
print(df.head(5))

# dropping nan values in datetime
df.dropna(subset=["Datetime"],inplace = True)

# remove unnecessary columns
df.drop(columns =["Date","Time"],inplace = True,axis =1)

df.set_index("Datetime",inplace = True)
print(df.head())

# remove negative numbers and convert it into nan
df.replace(-200,np.nan,inplace = True)
df.interpolate(method = "time",inplace = True)

print(df.head())

# feature engineering
df["hour"] = df.index.hour
df["month"] = df.index.month


print(df.head())
print(df.columns)

selected_columns = [ 'NO2(GT)','T','RH','AH','CO(GT)','hour','month']
data = df[selected_columns].copy()

print(data.head())
print(data.isnull().mean()*100)


corr = data.corr()
plt.figure()
sns.heatmap(corr,annot = True,cmap = "YlGnBu")
plt.title("Correlation Matrix")
plt.show()


plt.figure()
plt.plot(data["NO2(GT)"])
plt.show()


data.to_csv("air_quality_cleaned.csv")