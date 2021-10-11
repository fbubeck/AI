import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from time import time
import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import json


class TensorFlow():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0

    def train(self):
        # read config.json
        with open('config/config.json') as file:
            config = json.load(file)

        # Training Data (Preprocessing)
        xs_train = tf.convert_to_tensor(
            self.train_data[0], dtype=tf.int64)
        ys_train = tf.convert_to_tensor(
            self.train_data[1], dtype=tf.int64)

        # Exploration
        #plt.plot(xs_train, ys_train)
        # plt.show()

        # Training Parameters
        learning_rate = config["TensorFlow"]["learning_rate"]
        n_epochs = config["TensorFlow"]["n_epochs"]
        units = config["TensorFlow"]["n_units"]

        # Initializing Model
        self.model = keras.Sequential(
            [keras.layers.Dense(units=units, input_shape=[1])])

        # Define Optimizer
        opt = tf.keras.optimizers.Adam(lr=learning_rate)

        self.model.compile(
            optimizer=opt, loss='mean_squared_error')

        # Callback für TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch='500,520'
        )

        # Modelfitting
        start_training = time()
        history = self.model.fit(xs_train, ys_train, validation_split=0.33, epochs=n_epochs, callbacks=[
                                 tensorboard_callback])
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ TensorFlow ------')
        print(f'Duration Training: {duration_training} seconds')

        # summarize history for loss
        plt.plot(history.history['loss'], 'blue')
        plt.plot(history.history['val_loss'], 'red')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

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
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse

    # Parameter (zB Learning Rate (min-max in 10 Schritten), Anzahl Layer, Anzahl Epochen) aus JSON Konfigurationsfile laden
    # 1. Funktion: Training, return: Accuracy, Time
    # 2. Funktion Inferenz, return: Time
    # Varianz-Bias Trade off
    # Funktion in TensorFlow: Trainings, Testfehler ausgeben
    # Plots: Trainingsdaten, Testdaten, Accuracy vs Rechenzeit, Overfitting-plot
    # Weitere Modelle trainieren
    # Classcompliance
