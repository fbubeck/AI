from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time

start = time()

# Train Data
xs = [1, 2, 3]
ys = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]

# reshape Arrays
xs = np.matrix(xs).T.A
ys = np.matrix(ys).T.A
xs_test = np.matrix(xs_test).T.A

# Modelfitting
model = linear_model.LinearRegression()
model.fit(xs, ys)

# Predictions
y_pred = model.predict(xs_test)

print("Predictions for inputs 7, 8, 9: ")
print(y_pred)

end = time()
print(f'It took {end - start} seconds!')
