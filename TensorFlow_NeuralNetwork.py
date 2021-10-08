import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import numpy as np
from time import time
import torch
from sklearn.metrics import mean_squared_error
import SampleData


class TensorFlow():
    model = 0

    def __init__(self, array_length):
        self.train_data = SampleData.get_data(array_length)
        self.test_data = SampleData.get_data(array_length)

    def train():
        # Train Data
        xs_train = tf.convert_to_tensor(
            TensorFlow.train_data[0], dtype=tf.int64)
        ys_train = tf.convert_to_tensor(
            TensorFlow.train_data[1], dtype=tf.int64)

        # Initializing Model
        TensorFlow.model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=[1])])

        # Define Optimizer
        opt = tf.keras.optimizers.Adam(lr=0.001)

        TensorFlow.model.compile(optimizer=opt, loss='mean_squared_error')

        # Callback für TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="/logs{}",
            histogram_freq=1,
            profile_batch='500,520'
        )

        # Modelfitting
        start_training = time()
        TensorFlow.model.fit(xs_train, ys_train, epochs=50,
                             callbacks=[tensorboard_callback])
        end_training = time()

        print('--- Profiler ---')
        print(f'Duration Training: {end_training - start_training} seconds')

    def test():
       # Test Data
        xs_test = tf.convert_to_tensor(TensorFlow.test_data[0], dtype=tf.int64)
        ys_test = tf.convert_to_tensor(TensorFlow.test_data[1], dtype=tf.int64)

        start_test = time()
        y_pred = TensorFlow.model.predict(xs_test)
        end_test = time()

        # MSE (Mean Squarred Error)
        mse = mean_squared_error(ys_test, y_pred)
        print('--- Summary ---')
        print('MSE: ', mse)

        print(f'Duration Inferenz: {end_test - start_test} seconds')

    # Parameter (zB Learning Rate (min-max in 10 Schritten), Anzahl Layer, Anzahl Epochen) aus JSON Konfigurationsfile laden
    # 1. Funktion: Training, return: Accuracy, Time
    # 2. Funktion Inferenz, return: Time
    # Varianz-Bias Trade off
    # Funktion in TensorFlow: Trainings, Testfehler ausgeben
    # Plots: Trainingsdaten, Testdaten, Accuracy vs Rechenzeit, Overfitting-plot
    # Weitere Modelle trainieren
    # Classcompliance
