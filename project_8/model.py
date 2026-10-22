import torch
import torch.nn as nn
import torch.nn.functional as f


class lstm_fcn(nn.Module):
    def __init__(self,input_size=590,hidden=128,num_class = 2):
        super(lstm_fcn,self).__init__()
        self.lstm = nn.LSTM(input_size = 1,hidden_size=hidden,batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=8,padding=4)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,padding=2)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=64,kernel_size=3,padding=1)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden + 64,num_class)

    def forward(self,x):
        x = x.unsqueeze(2)

        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:,-1,:] 

        cnn_input = x.permute(0,2,1)
        x_cnn = f.relu(self.conv1(cnn_input))
        x_cnn = f.relu(self.conv2(x_cnn))
        x_cnn = f.relu(self.conv3(x_cnn))

        x_cnn = self.gap(x_cnn).squeeze(2)

        combined = torch.cat([lstm_feat,x_cnn],dim=1)
        out = self.fc(combined)

        return out
