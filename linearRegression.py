from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time

start = time()

# Train Data
xs_train = [1, 2, 3]
ys_train = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]
ys_test = [14, 16, 18]

# reshape Arrays
xs_train = np.matrix(xs_train).T.A
ys_train = np.matrix(ys_train).T.A
xs_test = np.matrix(xs_test).T.A
ys_test = np.matrix(ys_test).T.A

# Modelfitting
model = linear_model.LinearRegression()
model.fit(xs_train, ys_train)

# Predictions
y_pred = model.predict(xs_test)

print("Predictions for inputs 7, 8, 9: ")
print(y_pred)

# MSE
print("Mean squared error: %.2f" %
      np.mean((y_pred - ys_test) ** 2))

end = time()
print(f'It took {end - start} seconds!')
