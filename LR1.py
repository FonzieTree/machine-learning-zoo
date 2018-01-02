import numpy as np
import tensorflow as tf
X = np.arange(10000)
X = X.astype('float32')
Y = 2*X + 0.1*np.random.randn(10000)
Y = Y.astype('float32')
x  = tf.placeholder(tf.float32, name = 'x')
y = tf.placeholder(tf.float32, name = 'y')
w = tf.Variable(0.0, name = 'weights')
yp = w * x
loss = tf.reduce_mean(tf.abs(y - yp), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        result = sess.run(optimizer, feed_dict = {x:X, y:Y})
        print('w:	', sess.run(w), '	loss:	', sess.run(loss, feed_dict = {x:X, y:Y}))
