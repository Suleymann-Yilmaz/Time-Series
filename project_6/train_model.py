import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt

X_train = np.load("X_train_ls.npy")
X_test = np.load("X_test_ls.npy")
y_train = np.load("y_train_ls.npy")
y_test = np.load("y_test_ls.npy")

print(X_train.shape[2])

X_train = torch.tensor(X_train,dtype = torch.float32)
y_train = torch.tensor(y_train,dtype = torch.float32)

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
droupout = 0.2
output_size = 1
num_epoch = 100

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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

loss_list = []

for epoch in range(num_epoch):
    model.train()
    output=model(X_train)
    loss = criterion(output,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"{epoch + 1}, loss - {loss.item()}")


plt.figure()
plt.plot(loss_list)
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

torch.save(model.state_dict(),"lstm_model.pt")