import torch
from model import lstm_fcn
import torch.nn as nn
from preprocess import train_loader,test_loader
from tqdm import tqdm 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = lstm_fcn().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

num_epoch = 5

for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs,labels in tqdm(train_loader):
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss * inputs.size(0)
        _,predicted = torch.max(output,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss /total
    accuracy = correct / total

    print(f"Number of Epoch : {epoch+1}/{num_epoch} - Accuracy : {accuracy}")


torch.save(model.state_dict(),"lstm_fcn.pth")
print("Model saved successfully")