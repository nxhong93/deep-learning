import tensorflow  as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

n_steps = 28
n_inputs = 28
n_neurons = 100
n_layers = 3
n_outputs = 10
lr = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
mnist = input_data.read_data_sets('/tmp/data/')
x_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons) for layer in range(n_layers)]
multi_layer = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name = 'softmax')
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
loss = tf.reduce_mean(xentropy, name = 'lost')
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



init = tf.global_variables_initializer()

e = 10
batch = 150

with tf.Session() as sess:
	init.run()
	for epoch in range(e):
		for iteration in range(mnist.train.num_examples // batch):
			x_batch, y_batch = mnist.train.next_batch(batch)
			x_batch = x_batch.reshape((-1, n_steps, n_inputs))
			sess.run(training_op, feed_dict = {X: x_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict = {X: x_batch, y: y_batch})
		acc_tess = accuracy.eval(feed_dict = {X: x_test, y: y_test})
		print(epoch, acc_train, acc_tess)


