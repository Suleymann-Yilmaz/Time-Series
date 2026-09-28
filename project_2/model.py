import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# load X_train and y_train
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Create GRU Model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(units = 64,activation = "tanh"),
    tf.keras.layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              metrics = ["mae"])

history = model.fit(X_train,y_train,batch_size = 32,epochs = 10)


# Visualize the model's performance
plt.figure()
plt.plot(history.history["loss"],label = "Loss")
plt.title("Loss")
plt.legend()
plt.show()

model.save("gru_model.h5")