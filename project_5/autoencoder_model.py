import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,LSTM,RepeatVector,TimeDistributed,Dense
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load("X_train.npy")
print(X_train,X_train.shape)

timesteps = X_train.shape[1]
input_dim = X_train.shape[2]

inputs = Input(shape = (timesteps,input_dim))
encoded = LSTM(64,return_sequences = False)(inputs)
latent = RepeatVector(timesteps)(encoded)
decoded = LSTM(64,activation = "relu",return_sequences = True)(latent)
output = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs,output)
autoencoder.compile(loss = "mse",
                    optimizer=tf.keras.optimizers.Adam())

early_stop = EarlyStopping(
    monitor="loss",
    patience = 5,
    restore_best_weights = True
)

history = autoencoder.fit(X_train,X_train,epochs=50,batch_size=32,shuffle=True,callbacks=[early_stop])


plt.figure()
plt.plot(history.history["loss"],label = "Train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

autoencoder.save("autoencoder_lstm.h5")
print("We built the lstm autoencoder model successfully")