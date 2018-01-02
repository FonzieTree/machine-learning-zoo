import numpy as np
import tensorflow as tf
X = np.arange(100)
X = X.astype('float32')
Y = 2*X + 0.1*np.random.randn(100)
Y = Y.astype('float32')

class Graph:
    def __init__(self):
        self.graph = tf.Graph()
		with self.graph.as_default():
            self.x  = tf.placeholder(tf.float32, name = 'x')
            self.y = tf.placeholder(tf.float32, name = 'y')
            self.w = tf.Variable(0.0, name = 'weights')
            self.yp = self.w * self.x
            self.loss = tf.reduce_mean(tf.abs(self.y - self.yp), name = 'loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.loss)
g = Graph();
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        result = sess.run(g.optimizer, feed_dict = {g.x:X, g.y:Y})
        print(sess.run(w))
