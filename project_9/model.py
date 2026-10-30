import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,BatchNormalization,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

data_dir = "data"
X_train = np.load(os.path.join(data_dir,"X_train.npy"))
X_test = np.load(os.path.join(data_dir,"X_test.npy"))
y_train = np.load(os.path.join(data_dir,"y_train.npy"))
y_test = np.load(os.path.join(data_dir,"y_test.npy"))

X_train = X_train[...,np.newaxis]
X_test = X_test[...,np.newaxis]

model = Sequential()
model.add(Conv1D(filters =16,kernel_size = 5,activation = "relu",input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size = 2))
model.add(Conv1D(filters = 32,kernel_size = 5,activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size =2))
model.add(Conv1D(filters = 64,kernel_size = 3,activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),optimizer = "adam",metrics = ["accuracy"])

early_stop = EarlyStopping(monitor = "val_loss",restore_best_weights=True,patience=5)

history = model.fit(X_train,y_train,validation_split = 0.2,callbacks = [early_stop],batch_size = 32,epochs = 5)

model.summary()

plt.figure()
plt.plot(history.history["loss"],label = "Train - Loss")
plt.plot(history.history["val_loss"],label = "Validation - Loss")
plt.plot(history.history["accuracy"],label = "Train Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation - Accuracy")
plt.legend()
plt.tight_layout()
plt.grid("on")
plt.show()


model.save("eartquake_model.h5")