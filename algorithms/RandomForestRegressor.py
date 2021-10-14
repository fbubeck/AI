import sklearn
from sklearn.ensemble import RandomForestRegressor
from time import time
import numpy as np
from sklearn import metrics
import json


class RandomForest():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0

    def train(self):
        # read config.json
        with open('config/config.json') as file:
            config = json.load(file)

        # Training Parameters
        n_estimators = config["RandomForest"]["n_estimators"]
        random_state = config["RandomForest"]["random_state"]

        # Training Data
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.ravel(self.train_data[1])

        # Modelfitting
        RandomForest.model = RandomForestRegressor(n_estimators=n_estimators)
        start_training = time()
        RandomForest.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ RandomForest ------')
        print(f'Duration Training: {duration_training} seconds')

        return duration_training

    def test(self):
        # Test Data
        xs_test = np.matrix(self.test_data[0]).T.A
        ys_test = np.matrix(self.test_data[1]).T.A

        # Predictions
        start_test = time()
        y_pred = RandomForest.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = metrics.mean_squared_error(ys_test, y_pred)
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse
