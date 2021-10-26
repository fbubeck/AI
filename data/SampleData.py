
import random
import numpy as np


class SampleData():
    def __init__(self, array_length, min_bias, max_bias):
        self.array_length = array_length
        self.x_array = np.empty(self.array_length, dtype=object)
        self.y_array = np.empty(self.array_length, dtype=object)
        self.noice = np.empty(self.array_length, dtype=object)
        self.min_bias = min_bias
        self.max_bias = max_bias
        self.varianz = 0

    def get_Data(self):
        for x in range(0, self.array_length):
            IntRandom = random.randint(1, self.array_length)
            self.x_array[x] = IntRandom
            self.y_array[x] = IntRandom*2
            self.noice[x] = random.randint(self.min_bias, self.max_bias)

        # Varianz des Noice für Berechnung des MSE
        self.varianz = np.var(self.noice)

        for x in range(0, self.array_length):
            self.y_array[x] += self.noice[x]

        return self.x_array, self.y_array, self.varianz
