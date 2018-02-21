import tensorflow as tf

from encoders.encode_helper import EncodeHelper
from helper import FileHelper
from preprocess_data import PreprocessData


#mode = tf.estimator.ModeKeys.PREDICT

epochs = 2
alphabet_size = len(EncodeHelper.alphabet_standard)
learning_rate = 0.001


def cnn_conv_layer(data_tensor, name):
    # Input Layer
    input_layer = tf.reshape(data_tensor, [-1, 1024, alphabet_size, 1])
    tr_input_layer = tf.transpose(input_layer, perm=[0, 2, 1, 3])

    # setup the filter input shape for tf.nn.conv_2d
    filter_shape = [alphabet_size, 5]
    num_input_channels = 1
    num_filters = 16
    name = "conv_" + name
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    # Convolutional Layer #1
    out_layer = tf.nn.conv2d(tr_input_layer, filter=weights, strides=[1, alphabet_size, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    return out_layer


def max_pooling(in_layer):
    pool_shape = [1, 5]

    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 2, 1]
    out_layer = tf.nn.max_pool(in_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def full_connection(in_layer, count_neurons, name):
    name = "fc_" + name

    wd = tf.Variable(tf.truncated_normal([in_layer.shape[1].value, count_neurons], stddev=0.03), name=name + '_wd')
    bd = tf.Variable(tf.truncated_normal([count_neurons], stddev=0.01), name=name + 'bd')
    dense_layer = tf.matmul(in_layer, wd) + bd
    dense_layer = tf.nn.relu(dense_layer)

    return dense_layer


def make_flat(in_layer):
    return tf.reshape(in_layer, [-1, in_layer.shape[1].value * in_layer.shape[2].value * in_layer.shape[3].value])


tf.reset_default_graph()
tf.set_random_seed(111)
y = tf.placeholder(tf.float32, [None], name="y")
y_batch = tf.reshape(y, [-1, 1], name="y_batch")
x_batch = tf.placeholder(tf.float32, [None, 1024, alphabet_size], name="x_batch")

# t1 = tf.constant(encoded_messages[0:3])

conv1_layer = cnn_conv_layer(x_batch, name="1")
pool1_layer = max_pooling(conv1_layer)
flat_layer = make_flat(pool1_layer)
full_connected1 = full_connection(flat_layer, count_neurons=500, name="1")
full_connected2 = full_connection(full_connected1, count_neurons=100, name="2")
full_connected3 = full_connection(full_connected2, count_neurons=1, name="3")

n_result = tf.Print(full_connected3, [full_connected3], message="This is full_connected3: ")

diff = tf.subtract(n_result, y_batch)
error = tf.square(diff)
mean_square_error = tf.reduce_mean(error)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)



encoded_messages, scores = PreprocessData.get_encoded_messages()

#with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs=30
for epoch in range(epochs):
    result = sess.run(optimiser, feed_dict={x_batch: encoded_messages[0:3], y: scores[0:3]})

print("Result: {}".format(result))

mse = sess.run(mean_square_error, feed_dict={x_batch: encoded_messages[3:6], y: scores[3:6]})
print("Result: {}".format(mse))
