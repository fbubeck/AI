from sklearn import linear_model, metrics
import numpy as np
from matplotlib import pyplot as plt
from time import time


class LinearRegression():
    model = 0

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self):
        # Training Data
        self.xs_train = np.matrix(self.train_data[0]).T.A
        self.ys_train = np.matrix(self.train_data[1]).T.A

        # Modelfitting
        LinearRegression.model = linear_model.LinearRegression()
        start_training = time()
        LinearRegression.model.fit(self.xs_train, self.ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        print('------ LinearRegression ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Coefficients: ', LinearRegression.model.coef_)

        return duration_training

    def test(self):
        # Test Data
        self.xs_test = np.matrix(self.test_data[0]).T.A
        self.ys_test = np.matrix(self.test_data[1]).T.A

        # Predictions
        start_test = time()
        self.y_pred = LinearRegression.model.predict(self.xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = metrics.mean_squared_error(self.ys_test, self.y_pred)
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse

    def plot(self):
        plt.scatter(self.xs_train, self.ys_train, color='b', s=5)
        plt.plot(self.xs_train, self.y_pred, color='r')
        plt.title('Linear Regression Model')
        plt.ylabel('y (Train Data)')
        plt.xlabel('x (Train Data)')
        plt.savefig('plots/LinearRegression_Training-Model-Viz.png')
        plt.show()

        plt.scatter(self.xs_test, self.ys_test, color='b', s=5)
        plt.plot(self.xs_test, self.y_pred, color='r')
        plt.title('Linear Regression Model')
        plt.ylabel('y (Test Data)')
        plt.xlabel('x (Test Data)')
        plt.savefig('plots/LinearRegression_Test-Model-Viz.png')
        plt.show()
        print("Linear Regression Model Plot saved...")
        print("")
