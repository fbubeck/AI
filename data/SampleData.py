
import random
import numpy as np


class SampleData():
    def __init__(self, array_length, min_bias, max_bias):
        self.array_length = array_length
        self.x_array = np.empty(self.array_length, dtype=object)
        self.y_array = np.empty(self.array_length, dtype=object)
        self.min_bias = min_bias
        self.max_bias = max_bias
        # Klassenvariable mse

    def getData(self):
        for x in range(0, self.array_length):
            IntRandom = random.randint(1, self.array_length)
            self.x_array[x] = IntRandom
            self.y_array[x] = IntRandom*2 + \
                random.randint(
                    self.min_bias, self.max_bias)  # in separate Funktion auslagern und analysieren (mse)
            # Array [1000] mit Noice erzeugen und Varianz berechnen
            # Theoretisches Limit ist der Noice

        return self.x_array, self.y_array
