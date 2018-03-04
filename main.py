import tensorflow as tf
from neural_network import cnn
from encoders.encode_helper import EncodeHelper
from preprocess.preprocess_helper import PreprocessHelper

#run tensorboard:
#tensorboard --logdir=logs

#mode = tf.estimator.ModeKeys.PREDICT

# Initialize variables
VERBOSE = True
epochs = 1
batch_size = 1000
alphabet_size = len(EncodeHelper.alphabet_standard)
learning_rate = 0.001

# Initialize graph
tf.reset_default_graph()
tf.set_random_seed(111)

# Initialize inputs
y = tf.placeholder(tf.float32, [None], name="y")
y_batch = tf.reshape(y, [-1, 1], name="y_batch")
x_batch = tf.placeholder(tf.float32, [None, 1024, alphabet_size], name="x_batch")

x_batch_4 = tf.reshape(x_batch, [-1, 1024, alphabet_size, 1])
x_batch_input = tf.transpose(x_batch_4, perm=[0, 2, 1, 3])

# Convolutional-max pool 1
conv1_layer = cnn.cnn_conv_layer(x_batch_input, name="1",
                                 filter_shape=[alphabet_size, 5], filters=16, channels=1)
pool1_layer = cnn.max_pooling(conv1_layer, "1", pool_shape=[1, 5])

if VERBOSE:
    pool1_layer = tf.Print(pool1_layer, [conv1_layer, pool1_layer],
                           message="This is conv1_layer, pool1_layer: ")

# Convolutional-max pool 2
conv2_layer = cnn.cnn_conv_layer(pool1_layer, name="2",
                                 filter_shape=[1, 5], filters=16, channels=16)
pool2_layer = cnn.max_pooling(conv2_layer, "2", pool_shape=[1, 5])

if VERBOSE:
    pool2_layer = tf.Print(pool2_layer, [conv2_layer, pool2_layer],
                           message="This is conv2_layer, pool2_layer: ")

# Full connection
flat_layer = cnn.make_flat(pool2_layer)

full_connected1 = cnn.full_connection(flat_layer, count_neurons=500, name="1")
full_connected2 = cnn.full_connection(full_connected1, count_neurons=100, name="2")
full_connected3 = cnn.full_connection(full_connected2, count_neurons=1, name="3")

if VERBOSE:
    full_connected3 = tf.Print(full_connected3,
                               [full_connected1, full_connected2, full_connected3],
                               message="This is full_connected 1,2,3: ")

# Optimization
diff = tf.subtract(full_connected3, y_batch)
error = tf.square(diff)
mean_square_error = tf.reduce_mean(error)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

# Run
train_messages, train_scores = PreprocessHelper.get_encoded_messages("data/encoded/standard/train/Beauty_train_32343.json.pickle")
test_messages, test_scores = PreprocessHelper.get_encoded_messages("data/encoded/standard/test/Beauty_test_13862.json.pickle")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/graph', sess.graph)
    sess.run(tf.global_variables_initializer())

    start_index = 0
    end_index = batch_size
    train_length = len(train_messages)

    for epoch in range(epochs):
        while start_index < len(train_messages):
            if train_length < end_index:
                end_index = train_length

            print("Epoch {}, Batch {} - {}: ".format(epoch, start_index, end_index))
            result = sess.run(optimiser, feed_dict={x_batch: train_messages[start_index:end_index],
                                                    y: train_scores[start_index:end_index]})

            start_index += batch_size
            end_index += batch_size
            # print("Result: {}".format(result))

        mse = sess.run(mean_square_error, feed_dict={x_batch: test_messages[0:batch_size], y: test_scores[0:batch_size]})
        print("MSE: {}".format(mse))

    writer.close()


