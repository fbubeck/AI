import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
from time import time
import datetime
from sklearn.metrics import mean_squared_error
import json


class TensorFlow():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0
        self.varianz = 0

    def train(self):
        # read config.json
        with open('config/config.json') as file:
            config = json.load(file)

        # Training Data (Preprocessing)
        self.xs_train = tf.convert_to_tensor(
            self.train_data[0], dtype=tf.int64)
        self.ys_train = tf.convert_to_tensor(
            self.train_data[1], dtype=tf.int64)

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

        # Modeling
        start_training = time()
        self.history = self.model.fit(self.xs_train, self.ys_train, validation_split=0.33, verbose=1, epochs=n_epochs, callbacks=[
            tensorboard_callback])
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ TensorFlow ------')
        print(f'Duration Training: {duration_training} seconds')

        return duration_training

    def test(self):
       # Test Data (Preprocessing)
        self.xs_test = tf.convert_to_tensor(self.test_data[0], dtype=tf.int64)
        self.ys_test = tf.convert_to_tensor(self.test_data[1], dtype=tf.int64)
        self.varianz = self.test_data[2]

        # Predict Data
        start_test = time()
        self.y_pred = self.model.predict(self.xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE (Mean Squarred Error)
        mse = (self.varianz/mean_squared_error(self.ys_test, self.y_pred))-1
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse

    def plot(self):
        # Plot loss and val_loss
        px = 1/plt.rcParams['figure.dpi']  
        __fig = plt.figure(figsize=(800*px, 600*px))
        plt.plot(self.history.history['loss'], 'blue')
        plt.plot(self.history.history['val_loss'], 'red')
        plt.title('Neural Network Training loss history')
        plt.ylabel('loss (log scale)')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.savefig('plots/TensorFlow_Loss-Epochs-Plot.png')
        plt.show()
        print("TensorFlow loss Plot saved...")
        print("")
