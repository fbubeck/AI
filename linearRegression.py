from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time
import SampleData


class LinearRegression():
    model = 0

    def __init__(self, array_length):
        self.train_data = SampleData.get_data(array_length)
        self.test_data = SampleData.get_data(array_length)

    def train():

        # Train Data
        xs_train = np.matrix(LinearRegression.x_train).T.A
        ys_train = np.matrix(LinearRegression.y_train).T.A

        # Modelfitting
        LinearRegression.model = linear_model.LinearRegression()
        start_training = time()
        LinearRegression.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('--- Profiler ---')
        print(f'Duration Training: {duration_training} seconds')

    def test():
        # Test Data
        xs_test = np.matrix(LinearRegression.x_test).T.A
        ys_test = np.matrix(LinearRegression.y_test).T.A

        # Predictions
        start_test = time()
        y_pred = LinearRegression.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        # MSE
        print('--- Summary ---')
        mse = np.mean((y_pred - ys_test) ** 2)
        print("Mean squared error: %.2f" % mse)

        print('--- Profiler ---')
        print(f'Duration Inferenz: {duration_test} seconds')

        return duration_test, mse
