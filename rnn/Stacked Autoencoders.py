import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import sys


mnist = input_data.read_data_sets('/tmp/data/')


n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = 300
n_output = n_inputs
lr = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, [None, n_inputs])
with tf.contrib.framework.arg_scope(
[fully_connected],
		activation_fn=tf.nn.elu,
		weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
		weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
	hidden1 = fully_connected(X, n_hidden1)
	hidden2 = fully_connected(hidden1, n_hidden2)
	hidden3 = fully_connected(hidden2, n_hidden3)
	outputs = fully_connected(hidden3, n_output)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))	

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(lr)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch = 150

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		n_batches = mnist.train.num_examples//batch
		for iteration in range(n_batches):
			# print ('\r{}'.format(100*iteration//n_batches))
			sys.stdout.flush()
			x_batch, y_batch = mnist.train.next_batch(batch)
			sess.run(training_op, feed_dict = {X: x_batch})
		loss_train = reconstruction_loss.eval(feed_dict = {X: x_batch})
		print("\r{}".format(epoch), "Train MSE:", loss_train)
        # saver.save(sess, "./my_model_all_layers.ckpt")
def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

show_reconstructed_digits(X, outputs, "./my_model_all_layers.ckpt")
save_fig("reconstruction_plot")        