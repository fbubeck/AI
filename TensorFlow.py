import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import numpy as np
from time import time
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import SampleData


class TensorFlow():
    def __init__(self, array_length):
        self.sampleData = SampleData.SampleData(array_length)
        self.train_data = self.sampleData.get_Data()
        self.test_data = self.sampleData.get_Data()
        self.model = 0

    def train(self):
        # Training Data (Preprocessing)
        xs_train = tf.convert_to_tensor(
            self.train_data[0], dtype=tf.int64)
        ys_train = tf.convert_to_tensor(
            self.train_data[1], dtype=tf.int64)

        # Exploration
        plt.plot(xs_train, ys_train)
        plt.show()

        # Initializing Model
        self.model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=[1])])

        # Define Optimizer
        opt = tf.keras.optimizers.Adam(lr=0.001)

        self.model.compile(
            optimizer=opt, loss='mean_squared_error')

        # Callback für TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="/logs{}",
            histogram_freq=1,
            profile_batch='500,520'
        )

        # Modelfitting
        start_training = time()
        self.model.fit(xs_train, ys_train, epochs=50,
                       callbacks=[tensorboard_callback])
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ TensorFlow ------')
        print(f'Duration Training: {duration_training} seconds')

    def test(self):
       # Test Data (Preprocessing)
        xs_test = tf.convert_to_tensor(self.test_data[0], dtype=tf.int64)
        ys_test = tf.convert_to_tensor(self.test_data[1], dtype=tf.int64)

        # Predict Data
        start_test = time()
        y_pred = self.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE (Mean Squarred Error)
        mse = mean_squared_error(ys_test, y_pred)
        print('Mean squared error: ', mse)

        return duration_test, mse

    # Parameter (zB Learning Rate (min-max in 10 Schritten), Anzahl Layer, Anzahl Epochen) aus JSON Konfigurationsfile laden
    # 1. Funktion: Training, return: Accuracy, Time
    # 2. Funktion Inferenz, return: Time
    # Varianz-Bias Trade off
    # Funktion in TensorFlow: Trainings, Testfehler ausgeben
    # Plots: Trainingsdaten, Testdaten, Accuracy vs Rechenzeit, Overfitting-plot
    # Weitere Modelle trainieren
    # Classcompliance
