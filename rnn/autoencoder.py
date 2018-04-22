import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)

scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

lr = 0.01

X = tf.placeholder(tf.float32, [None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn = None)
outputs = fully_connected(hidden, n_outputs, activation_fn = None)

reconstrucion_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(lr)
training_op = optimizer.minimize(reconstrucion_loss)

init = tf.global_variables_initializer()

n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test})

fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.show()    



