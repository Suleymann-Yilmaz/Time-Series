import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import TensorDataset,DataLoader


df = pd.read_csv("data-secom.csv")
df.fillna(df.mean(),inplace = True)
print(df.isnull().mean())


df["label"] = df["label"].apply(lambda x : 0 if x == -1 else 1)


X = df.drop(columns = ["label"],axis = 1).values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled,y_resampled = smote.fit_resample(X_scaled,y)

X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42,stratify=y_resampled)

X_train_tensor = torch.tensor(X_train,dtype = torch.float32)
X_test_tensor = torch.tensor(X_test,dtype = torch.float32)
y_train_tensor = torch.tensor(y_train,dtype = torch.long)
y_test_tensor = torch.tensor(y_test,dtype = torch.long)


train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)