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
        # Train Data (Preprocessing)
        xs_train = tf.convert_to_tensor(
            TensorFlow.train_data[0], dtype=tf.int64)
        ys_train = tf.convert_to_tensor(
            TensorFlow.train_data[1], dtype=tf.int64)

        # Initializing Model
        TensorFlow.model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=[1])])

        # Define Optimizer
        opt = tf.keras.optimizers.Adam(lr=0.001)

        TensorFlow.model.compile(
            optimizer=opt, loss='mean_squared_error', metrics='mean_squared_error')

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

        # Time
        duration_training = end_training - start_training

        print('--- Profiler ---')
        print(f'Duration Training: {duration_training} seconds')

    def test():
       # Test Data (Preprocessing)
        xs_test = tf.convert_to_tensor(TensorFlow.test_data[0], dtype=tf.int64)
        ys_test = tf.convert_to_tensor(TensorFlow.test_data[1], dtype=tf.int64)

        # Evaluate Data
        print('--- Evaluation ---')
        results = TensorFlow.model.evaluate(xs_test, ys_test, batch_size=128)
        print('test loss, test acc:', results)

        # Predict Data
        start_test = time()
        y_pred = TensorFlow.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        # MSE (Mean Squarred Error)
        mse = mean_squared_error(ys_test, y_pred)
        print('--- Summary ---')
        print('MSE: ', mse)

        print('--- Profiler ---')
        print(f'Duration Inference: {duration_test} seconds')

        return duration_test, mse

    # Parameter (zB Learning Rate (min-max in 10 Schritten), Anzahl Layer, Anzahl Epochen) aus JSON Konfigurationsfile laden
    # 1. Funktion: Training, return: Accuracy, Time
    # 2. Funktion Inferenz, return: Time
    # Varianz-Bias Trade off
    # Funktion in TensorFlow: Trainings, Testfehler ausgeben
    # Plots: Trainingsdaten, Testdaten, Accuracy vs Rechenzeit, Overfitting-plot
    # Weitere Modelle trainieren
    # Classcompliance
