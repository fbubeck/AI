import sklearn
from sklearn.ensemble import RandomForestRegressor
from time import time
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


class RandomForest():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0
        self.varianz = self.test_data[2]

    def train(self):
        # read config.json
        with open('config/config.json') as file:
            config = json.load(file)

        # Training Parameters
        n_estimators = config["RandomForest"]["n_estimators"]
        random_state = config["RandomForest"]["random_state"]

        # Training Data
        self.xs_train = np.matrix(self.train_data[0]).T.A
        self.ys_train = np.ravel(self.train_data[1])

        # Modelfitting
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        start_training = time()
        self.model.fit(self.xs_train, self.ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        # Prediction for Training mse
        y_pred = self.model.predict(self.xs_train)

        mse = (mean_squared_error(self.ys_train, y_pred)/self.varianz)*100

        print('------ RandomForest ------')
        print(f'Duration Training: {duration_training} seconds')

        return duration_training, mse

    def test(self):
        # Test Data
        self.xs_test = np.matrix(self.test_data[0]).T.A
        self.ys_test = np.matrix(self.test_data[1]).T.A

        # Predictions
        start_test = time()
        y_pred = self.model.predict(self.xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = (mean_squared_error(self.ys_test, y_pred)/self.varianz)*100
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse, y_pred
