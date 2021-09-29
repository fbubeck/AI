import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error


# Start of Time Measuring
start = time()

# Initializing Model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Define Optimizer
opt = SGD(lr=0.1, momentum=0.9)

model.compile(optimizer=opt, loss='mean_squared_error')

# Train Data
xs_train = [1, 2, 3]
ys_train = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]
ys_test = [14, 16, 18]

# Callback für TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/logs{}",
    histogram_freq=1,
    profile_batch='500,520'
)

# Modelfitting
model.fit(xs_train, ys_train, epochs=150, callbacks=[tensorboard_callback])

# Predictions
y_pred = model.predict(xs_test)

print('Prediction for number 7:', y_pred[0])
print('Prediction for number 8:', y_pred[1])
print('Prediction for number 9:', y_pred[2])

# MSE (Mean Squarred Error)
mse = mean_squared_error(ys_test, y_pred)
print('MSE: ', mse)

# End of Time Measuring
end = time()
print('--- Profiler ---')
print(f'Duration: {end - start} seconds')
