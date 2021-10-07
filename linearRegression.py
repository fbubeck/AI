from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from time import time
from SampleData import x_train
from SampleData import y_train
from SampleData import x_test
from SampleData import y_test

# Train Data
xs_train = np.matrix(x_train).T.A
ys_train = np.matrix(y_train).T.A

# Test Data
xs_test = np.matrix(x_test).T.A
ys_test = np.matrix(y_test).T.A

# Modelfitting
model = linear_model.LinearRegression()
start = time()
model.fit(xs_train, ys_train)
end = time()

# Predictions
start2 = time()
y_pred = model.predict(xs_test)
end2 = time()

# MSE
print('--- Summary ---')
print("Mean squared error: %.2f" %
      np.mean((y_pred - ys_test) ** 2))

print('--- Profiler ---')
print(f'Duration Training: {end - start} seconds')
print(f'Duration Inferenz: {end2 - start2} seconds')
