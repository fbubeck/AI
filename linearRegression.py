from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time
import SampleData


class LinearRegression():
    model = 0

    def __init__(self, array_length):
        self.sampleData = SampleData.SampleData(array_length)
        self.train_data = self.sampleData.get_Data()
        self.test_data = self.sampleData.get_Data()

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
        mse = np.mean((y_pred - ys_test) ** 2)
        print("Mean squared error: %.2f" % mse)

        return duration_test, mse
