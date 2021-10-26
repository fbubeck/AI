from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np
from time import time


class DecisionTree():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0

    def train(self):
        # Training Data
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.matrix(self.train_data[1]).T.A

        self.model = DecisionTreeRegressor()

        # Modelfitting
        start_training = time()
        self.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ DecisionTree ------')
        print(f'Duration Training: {duration_training} seconds')

        return duration_training

    def test(self):
        # Test Data
        xs_test = np.matrix(self.test_data[0]).T.A
        ys_test = np.matrix(self.test_data[1]).T.A
        self.varianz = self.test_data[2]

        # Predictions
        start_test = time()
        y_pred = self.model.predict(xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = (self.varianz/metrics.mean_squared_error(ys_test, y_pred))-1
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse
