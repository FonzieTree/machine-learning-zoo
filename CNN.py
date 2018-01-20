import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(1)
epsilon = 1e-8
print('loading data')
data=np.genfromtxt(r"train.csv", dtype = 'int', skip_header=1, delimiter = ',')
print('finished loading')
x  = tf.placeholder(shape = (None, 28, 28, 1), dtype = tf.float32, name = 'x')
y = tf.placeholder(dtype = tf.int64, name = 'y')
batch_size = 100
data_size = len(data)
num_batches = int(len(data) / batch_size)


print('Building CNN model..')
# settings for optimization
training_epochs=100
p_keep_conv=0.8
p_keep_hidden=0.5

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

with tf.name_scope('layer1') as scope:
    # L1 Conv shape=(?, 28, 28, 32)
    #    Pool     ->(?, 14, 14, 32)
    L1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME'))
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, p_keep_conv)
with tf.name_scope('layer2') as scope:
    # L2 Conv shape=(?, 14, 14, 64)
    #    Pool     ->(?, 7, 7, 64)
    L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, p_keep_conv)
with tf.name_scope('layer3') as scope:
    # L3 Conv shape=(?, 7, 7, 128)
    #    Pool     ->(?, 4, 4, 128)
    #    Reshape  ->(?, 625)
    L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.reshape(L3, [-1, W4.get_shape().as_list()[0]])
    L3 = tf.nn.dropout(L3, p_keep_conv)
with tf.name_scope('layer4') as scope:
    # L4 FC 4x4x128 inputs -> 625 outputs
    L4 = tf.nn.relu(tf.matmul(L3, W4))
    L4 = tf.nn.dropout(L4, p_keep_hidden)

# Output(labels) FC 625 inputs -> 10 outputs
model = tf.matmul(L4, W5)
model = tf.nn.softmax(model, dim = -1, name = 'model')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model, labels = y), name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches-2):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1) * batch_size, data_size)
        batch_data = shuffled_data[start_index:end_index]
        xx = batch_data[:,1:]
        xx = xx.astype(np.float32)
        xx = xx/255 - 0.5
        xx = xx.reshape(-1, 28, 28, 1)
        yy = batch_data[:,0]
        result = sess.run(optimizer, feed_dict = {x:xx, y:yy})
    print('Epoch: ', i, '    loss: ', sess.run(loss, feed_dict = {x:xx, y:yy}))
    is_correct = tf.equal(tf.argmax(model, 1), y)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('Acc:', sess.run(accuracy, feed_dict={x:xx, y:yy}))
