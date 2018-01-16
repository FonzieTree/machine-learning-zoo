# Data source: https://www.kaggle.com/c/digit-recognizer/data
import numpy as np
import tensorflow as tf
np.random.seed(1)
epsilon = 1e-8
print('loading data')
data=np.genfromtxt(r"train.csv", dtype = 'int', skip_header=1, delimiter = ',')
print('finished loading')
ytrain = data[:,0]
xtrain = data[:,1:]/255
xtrain = xtrain - 0.5
xtrain = xtrain.reshape(-1, 28, 28, 1)
xtrain = xtrain.astype(np.float32)
x  = tf.placeholder(shape = (None, 28, 28, 1), dtype = tf.float32, name = 'x')
y = tf.placeholder(dtype = tf.int32, name = 'y')

batch_size = 4

#第一个卷积核
filter1 = tf.Variable(tf.random_normal([3,3,1,2]))
#第二个卷积核
filter2 = tf.Variable(tf.random_normal([3,3,2,8]))
#第三个卷积核
filter3 = tf.Variable(tf.random_normal([3,3,8,16]))

#第一层卷积
mat1 = tf.nn.conv2d(x, filter1, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(mat1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#第二层卷积
mat2 = tf.nn.conv2d(pool1, filter2, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(mat2)
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#第三层卷积
mat3 = tf.nn.conv2d(pool2, filter3, strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(mat3)
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#全连接层
W_fc1 = tf.truncated_normal([4*4*16, 10], stddev=0.1)
pool3_flat = tf.reshape(pool3, [-1, 4*4*16])
fc1 = tf.nn.relu(tf.matmul(pool3_flat, W_fc1))

yhat = tf.nn.softmax(fc1, dim = -1, name = 'yhat')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = yhat, labels = y), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        idx = np.random.randint(batch_size)]
        result = sess.run(optimizer, feed_dict = {x:xtrain, y:ytrain})
        print('round:   ', i, '         loss:   ', sess.run(loss, feed_dict = {x:xtrain, y:ytrain}))
