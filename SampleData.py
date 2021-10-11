import random
from random import randint
import numpy as np
from numpy import testing


class SampleData():
    def __init__(self, array_length):
        self.array_length = array_length
        self.x_array = np.empty(self.array_length, dtype=object)
        self.y_array = np.empty(self.array_length, dtype=object)

    def get_Data(self):
        for x in range(0, self.array_length):
            random = randint(1, self.array_length)
            self.x_array[x] = random
            self.y_array[x] = random*2 + randint(-10, 10)

        return self.x_array, self.y_array

        # To Do
        # Variable für Länge der Zahlenreihe, für Anzahl der Punkte;
        # Funktion zum Ausführen (Calculate)
        # 1. Instanz Training, 2. Instanz Test
        # Get Data Funktion (Konstruktor: Anzahl Zahlenreihe)
