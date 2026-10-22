import torch
import torch.nn as nn
from model import lstm_fcn
from preprocess import test_loader
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import seaborn as sns
import matplotlib.pyplot as plt


device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

model = lstm_fcn().to(device)
model.load_state_dict(torch.load("lstm_fcn.pth"))
model.eval()

all_prediction = []
all_labels = []

with torch.no_grad():
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _,predicted = torch.max(outputs,1)
        all_prediction.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())



acc = accuracy_score(all_labels,all_prediction)
f1 = f1_score(all_labels,all_prediction)

print(acc)
print(f1)
print(classification_report(all_labels,all_prediction))

cm = confusion_matrix(all_labels,all_prediction)

plt.figure()
sns.heatmap(cm,annot = True,fmt="d",xticklabels=  [0,1],yticklabels= [0,1])
plt.show()