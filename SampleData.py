import random
from random import randint
import numpy as np
from numpy import testing


class SampleData(object):
    # Initializing Arrays
    x_array = np.empty(50000, dtype=object)
    y_array = np.empty(50000, dtype=object)

    def __init__(self):
        pass

    def calculate_data(array_length):
        # Filling Arrays with random integers
        for x in range(0, array_length):
            random = randint(1, array_length)
            SampleData.x_array[x] = random
            SampleData.y_array[x] = random*2 + randint(-10, 10)

    def get_Data(array_length):
        SampleData.array_length = array_length
        SampleData.calculate_data(SampleData.array_length)

        return SampleData.x_array, SampleData.y_array

        # Variable für Länge der Zahlenreihe, für Anzahl der Punkte;
        # Funktion zum Ausführen (Calculate)
        # 1. Instanz Training, 2. Instanz Test
        # Get Data Funktion (Konstruktor: Anzahl Zahlenreihe)
