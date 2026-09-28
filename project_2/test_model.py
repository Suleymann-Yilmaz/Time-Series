import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import joblib
import tensorflow as tf

# load gru model
gru_model = tf.keras.models.load_model("gru_model.h5")

# load test set
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# make predictions with gru-model
y_pred = gru_model.predict(X_test)
print(y_pred,y_pred.shape,y_test,y_test.shape)




scaler_y = joblib.load("scaler_y.save")
# 
y_preds_inverse = scaler_y.inverse_transform(y_pred.reshape(-1,1))
y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1,1))

mae = mean_absolute_error(y_test_inverse,y_preds_inverse)
mse = mean_squared_error(y_test_inverse,y_preds_inverse)
print(f"Mean Absolute error : {mae},mean squared error : {mse}")

plt.figure()
plt.plot(y_test_inverse[:200],label = "True Test Values",c = "red",linestyle = "--")
plt.plot(y_preds_inverse[:200],c = "blue",label = "Predicted values")
plt.legend()
plt.show()