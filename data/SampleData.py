import random
from random import randint
import numpy as np


class SampleData():
    def __init__(self, array_length, min_bias, max_bias):
        self.array_length = array_length
        self.x_array = np.empty(self.array_length, dtype=object)
        self.y_array = np.empty(self.array_length, dtype=object)
        self.min_bias = min_bias
        self.max_bias = max_bias

    def get_Data(self):
        for x in range(0, self.array_length):
            random = randint(1, self.array_length)
            self.x_array[x] = random
            self.y_array[x] = random*2 + randint(self.min_bias, self.max_bias)

        return self.x_array, self.y_array

        # To Do
        # Variable für Länge der Zahlenreihe, für Anzahl der Punkte;
        # Funktion zum Ausführen (Calculate)
        # 1. Instanz Training, 2. Instanz Test
        # Get Data Funktion (Konstruktor: Anzahl Zahlenreihe)