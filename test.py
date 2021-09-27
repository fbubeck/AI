import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time

start = time()

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Train Data
xs_train = [1, 2, 3]
ys_train = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]
ys_test = [14, 16, 18]

# Callback für TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/logs{}", histogram_freq=1)

# Modelfitting
model.fit(xs_train, ys_train, epochs=1000, callbacks=[tensorboard_callback])

# Predictions
y_pred = model.predict(xs_test)
print("Predictions for inputs 7, 8, 9: ")
print(y_pred)

# MSE
print("Mean squared error: %.2f" %
      np.mean((y_pred - ys_test) ** 2))

end = time()
print(f'It took {end - start} seconds!')
