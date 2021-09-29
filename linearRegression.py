from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time

# Start of Time Measuring
start = time()

# Train Data
xs_train = [1, 2, 3]
ys_train = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]
ys_test = [14, 16, 18]

# reshape 1D Arrays to 2D Arrays
xs_train = np.matrix(xs_train).T.A
ys_train = np.matrix(ys_train).T.A
xs_test = np.matrix(xs_test).T.A
ys_test = np.matrix(ys_test).T.A

# Modelfitting
model = linear_model.LinearRegression()
model.fit(xs_train, ys_train)

# Predictions
y_pred = model.predict(xs_test)

print('Prediction for number 7:', y_pred[0])
print('Prediction for number 8:', y_pred[1])
print('Prediction for number 9:', y_pred[2])

# MSE
print("Mean squared error: %.2f" %
      np.mean((y_pred - ys_test) ** 2))

# End of Time Measuring
end = time()
print('--- Profiler ---')
print(f'Duration: {end - start} seconds')
