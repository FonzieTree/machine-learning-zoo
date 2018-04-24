# Data source: https://www.kaggle.com/c/digit-recognizer/data
# framework source: http://cs231n.github.io/convolutional-networks/
import numpy as np
import tensorflow as tf
np.random.seed(1)
epsilon = 1e-8
print('loading data')
data=np.genfromtxt(r"E:\mnist\mnist\train.csv", dtype = 'int', skip_header=1, delimiter = ',')
print('finished loading')
ytrain = data[:,0]
xtrain = data[:,1:]/255
xtrain = xtrain - 0.5
xtrain = xtrain.reshape(-1, 28, 28, 1)
xtrain = xtrain.astype(np.float32)
x  = tf.placeholder(shape = (None, 28, 28, 1), dtype = tf.float32, name = 'x')
y = tf.placeholder(dtype = tf.int32, name = 'y')
batch_size = 4
round = int(xtrain.shape[0]/batch_size)

net = tf.nn.conv2d(x, tf.Variable(tf.random_normal([3,3,1,64])), strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.conv2d(net, tf.Variable(tf.random_normal([3,3,64,64])), strides=[1, 1, 1, 1], padding='SAME')
net += tf.nn.relu(net)
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
net = tf.nn.local_response_normalization(net)

net = tf.nn.conv2d(net, tf.Variable(tf.random_normal([3,3,64,128])), strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.conv2d(net, tf.Variable(tf.random_normal([3,3,128,128])), strides=[1, 1, 1, 1], padding='SAME')
net += tf.nn.relu(net)
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
net = tf.nn.local_response_normalization(net)

net = tf.nn.conv2d(net, tf.Variable(tf.random_normal([3,3,128,256])), strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.conv2d(net, tf.Variable(tf.random_normal([3,3,256,256])), strides=[1, 1, 1, 1], padding='SAME')
net += tf.nn.relu(net)
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
net = tf.nn.local_response_normalization(net)

W_fc1 = tf.truncated_normal([4*4*256, 10], stddev=0.1)
net = tf.reshape(net, [-1, 4*4*256])
fc1 = tf.nn.relu(tf.matmul(net, W_fc1))

yhat = tf.nn.softmax(fc1, dim = -1, name = 'yhat')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = yhat, labels = y), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(41999):
        xx = xtrain[i:i+1]
        yy = ytrain[i:i+1]
        result = sess.run(optimizer, feed_dict = {x:xx, y:yy})
        print('round:   ', i, '         loss:   ', sess.run(loss, feed_dict = {x:xx, y:yy}))
