import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch.nn as nn
import torch


X_test = np.load("X_test_ls.npy")
y_test = np.load("y_test_ls.npy")

print(y_test.shape)

X_test = torch.tensor(X_test,dtype = torch.float32)
y_test = torch.tensor(y_test,dtype= torch.float32).view(-1,1)
X_train = np.load("X_train_ls.npy")

print(y_test.shape)

scaler = joblib.load("scaler_air.save")

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
droupout = 0.2
output_size = 1

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        output,_ = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc(output)
        return output

model = LSTMModel(input_size,hidden_size,num_layers,output_size,droupout)
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval()

with torch.no_grad():
    predictions = model(X_test)

y_preds = predictions.numpy()
y_true = y_test.numpy()

dummy_pred = np.zeros((len(y_preds),scaler.n_features_in_))
dummy_pred[:,0] = y_preds[:,0]

dummy_true = np.zeros((len(y_true),scaler.n_features_in_))
dummy_true[:,0] = y_true[:,0]

inv_y_preds = scaler.inverse_transform(dummy_pred)[:,0]
inv_y_true = scaler.inverse_transform(dummy_true)[:,0]


plt.figure()
plt.plot(inv_y_preds,label = "Predictions",c = "red",alpha = 0.5)
plt.plot(inv_y_true,label = "True", c = "blue")
plt.legend()
plt.show()