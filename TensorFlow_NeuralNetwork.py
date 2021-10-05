import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import numpy as np
from time import time
import torch
from sklearn.metrics import mean_squared_error
from SampleData import x_train
from SampleData import y_train
from SampleData import x_test
from SampleData import y_test

# Train Data
xs_train = x_train
ys_train = y_train

# Test Data
xs_test = x_test
ys_test = x_test

xs_train = np.asarray(xs_train).astype(np.float32)
ys_train = np.asarray(ys_train).astype(np.float32)
xs_test = np.asarray(xs_test).astype(np.float32)
ys_test = np.asarray(ys_test).astype(np.float32)


# Initializing Model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Define Optimizer
opt = SGD(lr=0.1, momentum=0.9)

model.compile(optimizer=opt, loss='mean_squared_error')

# Callback für TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/logs{}",
    histogram_freq=1,
    profile_batch='500,520'
)

# Modelfitting
start = time()
model.fit(xs_train, ys_train, epochs=150, callbacks=[tensorboard_callback])
end = time()

# Predictions
start2 = time()
y_pred = model.predict(xs_test)
end2 = time()

# MSE (Mean Squarred Error)
mse = mean_squared_error(ys_test, y_pred)
print('--- Summary ---')
print('MSE: ', mse)

print('--- Profiler ---')
print(f'Duration Training: {end - start} seconds')
print(f'Duration Inferenz: {end2 - start2} seconds')
