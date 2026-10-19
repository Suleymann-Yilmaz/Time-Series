import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_incep
import tensorflow as tf

X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train_scaled.npy")
y_test = np.load("y_test_scaled.npy")

input_shape = (X_train.shape[1],X_train.shape[2])
num_class = y_train.shape[1]

model = build_incep(input_shape,num_class)
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),metrics = ["accuracy"])

early_stop = EarlyStopping(monitor = "val_loss",patience=10,restore_best_weights=True)
check = ModelCheckpoint("best_model_inc.h5",monitor="val_loss",save_best_only=True)

history = model.fit(X_train,y_train,validation_split=0.2,epochs = 30,batch_size=64,callbacks=[early_stop,check],verbose=1)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label ="Model loss")
plt.plot(history.history["val_loss"],label = "Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label ="Trainin Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()