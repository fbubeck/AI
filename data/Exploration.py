from matplotlib import pyplot as plt
import numpy as np


class Exploration():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def analyseData(self):
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.matrix(self.train_data[1]).T.A

        xs_train_sorted = np.sort(xs_train)
        ys_train_sorted = np.sort(ys_train)

        fig, axs = plt.subplots(2)
        fig.suptitle('Training Data (Input/Output)')
        axs[0].plot(xs_train_sorted, 'blue')
        axs[1].plot(ys_train_sorted, 'red')
        plt.savefig('plots/DataExploration.png')
