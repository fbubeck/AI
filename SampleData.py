import random
from random import randint
import numpy as np

# Initializing Arrays with length = 500
x_train = np.empty(500, dtype=object)
y_train = np.empty(500, dtype=object)
x_test = np.empty(500, dtype=object)
y_test = np.empty(500, dtype=object)

# Filling Arrays with random integers
for x in range(0, 500):
    random = randint(1, 500)
    x_train[x] = random
    y_train[x] = random*2
    random2 = randint(1, 500)
    x_test[x] = random
    y_test[x] = random*2
    random = 0
    random2 = 0

print(type(x_train))
