import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix
import os

model = load_model("eartquake_model.h5")
data_dir = "data"
X_test = np.load(os.path.join(data_dir,"X_test.npy"))
y_test = np.load(os.path.join(data_dir,"y_test.npy"))

X_test = X_test[...,np.newaxis]


y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba>0.5).astype("int")
accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test,y_pred)

print("Accuracy",accuracy)
print("f1",f1)
print("recall",recall)
print("confusion matrix",confusion)
print("precision",precision)
