from matplotlib import pyplot as plt
import numpy as np


class Exploration():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def analyseData(self):
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.matrix(self.train_data[1]).T.A

        plt.plot(xs_train)
        plt.ylim(-10000, 100000)
        plt.xlim(-10000, 100000)
        plt.show()

        plt.plot(ys_train)
        plt.ylim(-10000, 100000)
        plt.xlim(-10000, 100000)
        plt.show()
