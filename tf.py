import tensorflow  as tf
import numpy as np



def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

m1 = [[1.0, 2.0], [3.0, 4.0]]

m2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

m3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
t1 = tf.convert_to_tensor(m1, dtype=tf.float32)

neg_x = tf.negative(m2)


with tf.Session() as sess:
    result = sess.run(neg_x)
    print(result)

sess = tf.InteractiveSession()
result = neg_x.eval()
print (result)


raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]

spikes = tf.Variable([False] * len(raw_data), name='spikes')
spikes.initializer.run()
saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()
  
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
print (f)

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

reset_graph()



