import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
data = np.array([np.arange(100) + 0.1 * np.random.randn(100), 2 * np.arange(100)]).T
n_samples = data.shape[1]
X = tf.placeholder(dtype = tf.float32, name = 'X')
Y = tf.placeholder(dtype = tf.float32, name = 'Y')
w = tf.Variable(0.0, name = 'w')
b = tf.Variable(0.0, name = 'b')
Y_predicted = X * w + b
loss = tf.square(Y - Y_predicted, name = 'loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
with tf.Session() as sess:
    # initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    for i in range(10):
	    total_loss = 0.0
	    for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict = {X: x, Y: y})
            total_loss += l
	    print("Epoch {0}: {1}".format(i, total_loss/n_samples))
    w, b = sess.run([w, b])
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
