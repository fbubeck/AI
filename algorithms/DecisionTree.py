from sklearn.tree import DecisionTreeRegressor
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

class DecisionTree():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = 0
        self.varianz = self.test_data[2]

    def train(self):
        # Training Data
        self.xs_train = np.matrix(self.train_data[0]).T.A
        self.ys_train = np.matrix(self.train_data[1]).T.A

        self.model = DecisionTreeRegressor()

        # Modelfitting
        start_training = time()
        self.model.fit(self.xs_train, self.ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        # Prediction for Training mse
        y_pred = self.model.predict(self.xs_train)

        mse = (mean_squared_error(self.ys_train, y_pred)/self.varianz)*100

        print('------ DecisionTree ------')
        print(f'Duration Training: {duration_training} seconds')

        return duration_training, mse

    def test(self):
        # Test Data
        self.xs_test = np.matrix(self.test_data[0]).T.A
        self.ys_test = np.matrix(self.test_data[1]).T.A

        # Predictions
        start_test = time()
        self.y_pred = self.model.predict(self.xs_test)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        # MSE
        mse = (mean_squared_error(self.ys_test, self.y_pred)/self.varianz)*100
        print("Mean squared error: %.2f" % mse)
        print("")

        return duration_test, mse

    def plot(self):
        px = 1/plt.rcParams['figure.dpi']  
        __fig = plt.figure(figsize=(800*px, 600*px))
        plt.scatter(self.xs_test, self.ys_test, color='b', s=1, label="Data", alpha=0.5)
        plt.scatter(self.xs_test, self.y_pred, color='r', s=1, label="Best fit", alpha=0.5)
        plt.title('Decision Tree Model')
        plt.ylabel('y (Output)')
        plt.xlabel('x (Input)')
        plt.legend()
        plt.savefig('plots/DecisionTree_Test-Model-Viz.png')
        plt.show()
        print("Decision Tree Model Plot saved...")
        print("")

