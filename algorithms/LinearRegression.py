from sklearn import linear_model, metrics
import numpy as np
from time import time


class LinearRegression():
    model = 0

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self):
        # Training Data
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.matrix(self.train_data[1]).T.A

        # Modelfitting
        LinearRegression.model = linear_model.LinearRegression()
        start_training = time()
        LinearRegression.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ LinearRegression ------')
        print(f'Duration Training: {duration_training} seconds')

    def test(self):
        # Test Data
        xs_test = np.matrix(self.test_data[0]).T.A
        ys_test = np.matrix(self.test_data[1]).T.A

        # Predictions
        start_test = time()
        y_pred = LinearRegression.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = metrics.mean_squared_error(ys_test, y_pred)
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse
