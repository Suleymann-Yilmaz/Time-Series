"""
> ## Plan 
>> ### 1. Load Data and Explore
>> ### 2. Preprocessing
>> ### 3. Model
>> ### 4. Evaluate
>> ### 5. Create an app

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv.gz")
print(data.head())
print(data.info())
print(data.describe())

data["date_time"] = pd.to_datetime(data["date_time"])
data.set_index("date_time",inplace = True)
print(data.head())

# visualize time-series
plt.figure()
plt.plot(data["traffic_volume"],label = "Traffic Volume")
plt.title("Traffic Volume Time - Series")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid()
plt.show()

data["hour"] = data.index.hour

hourly_avg = data.groupby("hour")["traffic_volume"].mean()

plt.figure()
sns.barplot(x = hourly_avg.index,y = hourly_avg.values,palette="viridis")
plt.title("Hourly Average Traffic Volume")
plt.show()