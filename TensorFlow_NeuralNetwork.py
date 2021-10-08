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
xs_train = tf.convert_to_tensor(x_train, dtype=tf.int64)
ys_train = tf.convert_to_tensor(y_train, dtype=tf.int64)

# Test Data
xs_test = tf.convert_to_tensor(x_test, dtype=tf.int64)
ys_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

# Initializing Model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Define Optimizer
opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=opt, loss='mean_squared_error')

# Callback für TensorBoard
""" tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/logs{}",
    histogram_freq=1,
    profile_batch='500,520'
) """

# Modelfitting
start = time()
model.fit(xs_train, ys_train, epochs=50)
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

# Parameter (zB Learning Rate (min-max in 10 Schritten), Anzahl Layer, Anzahl Epochen) aus JSON Konfigurationsfile laden
# 1. Funktion: Training, return: Accuracy, Time
# 2. Funktion Inferenz, return: Time
# Varianz-Bias Trade off
# Funktion in TensorFlow: Trainings, Testfehler ausgeben
# Plots: Trainingsdaten, Testdaten, Accuracy vs Rechenzeit, Overfitting-plot
# Weitere Modelle trainieren
# Classcompliance
