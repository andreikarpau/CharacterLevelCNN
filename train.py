import tensorflow as tf

from helpers.file_helper import FileHelper
from neural_network import cnn
from encoders.encode_helper import EncodeHelper
from helpers.preprocess_helper import PreprocessHelper

#run tensorboard:
#tensorboard --logdir=logs

#data
encoding_name = "standard"

# train_messages, train_scores = PreprocessHelper.get_encoded_messages_from_folder(
#     "data/encoded/{}/train".format(encoding_name))
# test_messages, test_scores = PreprocessHelper.get_encoded_messages_from_folder(
#     "data/encoded/{}/test".format(encoding_name))

train_messages, train_scores = PreprocessHelper.get_encoded_messages(
    "data/encoded/{}/train/Beauty_train_32343.json.pickle".format(encoding_name))
test_messages, test_scores = PreprocessHelper.get_encoded_messages(
    "data/encoded/{}/test/Beauty_test_13862.json.pickle".format(encoding_name))

#logging
logger = FileHelper.get_file_console_logger(encoding_name, "train.log")

# Initialize variables
VERBOSE = False
epochs = 1000
batch_size = 100
alphabet_size = len(EncodeHelper.alphabet_standard)
learning_rate = 0.0001
dropout = 0.25
mode = tf.estimator.ModeKeys.TRAIN


is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

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
                                 filter_shape=[alphabet_size, 5], filters=128, channels=1)
pool1_layer = cnn.max_pooling(conv1_layer, "1", pool_shape=[1, 5])

if VERBOSE:
    pool1_layer = tf.Print(pool1_layer, [conv1_layer, pool1_layer],
                           message="This is conv1_layer, pool1_layer: ")

# Convolutional-max pool 2
conv2_layer = cnn.cnn_conv_layer(pool1_layer, name="2",
                                 filter_shape=[1, 5], filters=256, channels=128)
pool2_layer = cnn.max_pooling(conv2_layer, "2", pool_shape=[1, 5])

if VERBOSE:
    pool2_layer = tf.Print(pool2_layer, [conv2_layer, pool2_layer],
                           message="This is conv2_layer, pool2_layer: ")

# Convolutional-max pool 3
conv3_layer = cnn.cnn_conv_layer(pool2_layer, name="3",
                                 filter_shape=[1, 3], filters=512, channels=256)
pool3_layer = cnn.max_pooling(conv3_layer, "3", pool_shape=[1, 3])

if VERBOSE:
    pool3_layer = tf.Print(pool3_layer, [conv3_layer, pool3_layer],
                           message="This is conv3_layer, pool3_layer: ")

# Full connection
# xxxxxflat_layer = cnn.make_flat(pool2_layer)
flat_layer = cnn.make_flat(pool3_layer)

dropout_layer = tf.layers.dropout(flat_layer, rate=dropout, training=is_training)

full_connected1 = cnn.full_connection(dropout_layer, count_neurons=500, name="1")
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
root_mean_square_error = tf.sqrt(mean_square_error)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

# Run
saver = tf.train.Saver()

config = tf.ConfigProto(log_device_placement=VERBOSE)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter('logs/graph/{}/'.format(encoding_name), sess.graph)
    sess.run(tf.global_variables_initializer())

    train_length = len(train_messages)

    for epoch in range(epochs):
        start_index = 0
        end_index = batch_size

        while start_index < train_length:
            if train_length < end_index:
                end_index = train_length

            if VERBOSE:
                logger.info("Epoch {}, Batch {} - {}: ".format(epoch, start_index, end_index))

            result = sess.run(optimiser, feed_dict={x_batch: train_messages[start_index:end_index],
                                                    y: train_scores[start_index:end_index]})

            batch_rmse = sess.run(root_mean_square_error,  feed_dict={x_batch: train_messages[start_index:end_index],
                                                    y: train_scores[start_index:end_index]})
            logger.info("Epoch {}, Batch {} - {}: RMSE = {}".format(epoch, start_index, end_index, batch_rmse))

            start_index += batch_size
            end_index += batch_size

        # rmse = sess.run(root_mean_square_error, feed_dict={x_batch: test_messages[0:batch_size], y: test_scores[0:batch_size]})
        # mse_message = "Epoch {} RMSE: {}".format(epoch, rmse)
        # logger.info(mse_message)
        #
        # saver.save(sess, "./logs/{}/model_epoch{}.ckpt".format(encoding_name, epoch))

    writer.close()


