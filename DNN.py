
# Data source: https://www.kaggle.com/c/digit-recognizer/data
import numpy as np
import tensorflow as tf
np.random.seed(1)
epsilon = 1e-8
print('loading data')
data=np.genfromtxt("train.csv", dtype = 'int', skip_header=1, delimiter = ',')
print('finished loading')
ytrain = data[:,0]
xtrain = data[:,1:]/255
xtrain = xtrain - 0.5
x  = tf.placeholder(shape = (None, 784), dtype = tf.float32, name = 'x')
y = tf.placeholder(dtype = tf.int32, name = 'y')
w1 = tf.Variable((tf.random_normal(shape = (784, 128), mean = 0.0, stddev = 0.1, dtype = tf.float32, seed = None, name = 'w1')))
w2 = tf.Variable((tf.random_normal(shape = (128, 64), mean = 0.0, stddev = 0.1, dtype = tf.float32, seed = None, name = 'w2')))
w3 = tf.Variable((tf.random_normal(shape = (64, 10), mean = 0.0, stddev = 0.1, dtype = tf.float32, seed = None, name = 'w3')))
mat1 = tf.matmul(x, w1, name = 'mat1')
relu1 = tf.nn.relu(mat1, name = 'relu1')
mat2 = tf.matmul(relu1, w2, name = 'mat2')
relu2 = tf.nn.relu(mat2, name = 'relu2')
mat3 = tf.matmul(relu2, w3, name = 'mat3')
yhat = tf.nn.softmax(mat3, dim = -1, name = 'yhat')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = yhat, labels = y), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        result = sess.run(optimizer, feed_dict = {x:xtrain, y:ytrain})
        print('round:   ', i, '         loss:   ', sess.run(loss, feed_dict = {x:xtrain, y:ytrain}))                                                                                                  
