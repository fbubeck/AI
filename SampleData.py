import random
from random import randint
import numpy as np

# Initializing Arrays with length = 5000
x_train = np.empty(5000, dtype=object)
y_train = np.empty(5000, dtype=object)
x_test = np.empty(5000, dtype=object)
y_test = np.empty(5000, dtype=object)

# Filling Arrays with random integers
for x in range(0, 5000):
    random = randint(1, 5000)
    x_train[x] = random
    y_train[x] = random*2
    random2 = randint(1, 5000)
    x_test[x] = random
    y_test[x] = random*2
    random = 0
    random2 = 0