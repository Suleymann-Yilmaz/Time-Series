import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import load_model

X_test = np.load("X_test_scaled.npy")
y_test = np.load("y_test_scaled.npy")
model = load_model("best_model_inc.h5")

y_predict_proba = model.predict(X_test)
y_predict = np.argmax(y_predict_proba,axis = 1)
y_true_classes = np.argmax(y_test,axis = 1)

activity_labels = ["walking","walking_upstairs","walking_downstairs","sitting","standing","laying"]


print(classification_report(y_true = y_true_classes,y_pred=y_predict,target_names=activity_labels))


cm = confusion_matrix(y_predict,y_true_classes)
plt.figure()
sns.heatmap(cm,annot=True,xticklabels=activity_labels,yticklabels=activity_labels)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()