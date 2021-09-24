import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Train Data
xs = [1, 2, 3]
ys = [2, 4, 6]

# Test Data
xs_test = [7, 8, 9]

# Callback für TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/logs{}", histogram_freq=1)

# Modelfitting
model.fit(xs, ys, epochs=1000, callbacks=[tensorboard_callback])

# Print Results
print(model.predict(xs_test))
