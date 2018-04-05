import math
import os
import tensorflow as tf

from helpers.file_helper import FileHelper
from neural_network import cnn
from encoders.encode_helper import EncodeHelper
from helpers.preprocess_helper import PreprocessHelper
from neural_network.cnn import CNNRunner


# run tensorboard:
# tensorboard --logdir=log

# Initialize variables
VERBOSE = False

alphabet_size = 0

use_whole_dataset = bool(os.getenv('USE_WHOLE_DATASET', 'False'))
data_path = os.getenv('DATA_PATH', 'data/encoded')

# current options: 'standard', 'standard_group'
encoding_name = os.getenv('ENCODING_NAME', 'standard_group')
output_postfix = os.getenv('OUTPUT_POSTFIX', 'run_1')
output_folder = os.getenv('OUTPUT_FOLDER', 'output')

if encoding_name == "standard":
    alphabet_size = len(EncodeHelper.alphabet_standard)
elif encoding_name == "standard_group":
    alphabet_size = len(EncodeHelper.make_standart_group_encoding()['a'])

full_output_name = "{}_{}".format(encoding_name, output_postfix)

epochs = int(os.getenv('EPOCHS_COUNT', 1000))
batch_size = int(os.getenv('BATCH_SIZE', 100))
learning_rate = float(os.getenv('LEARNING_RATE', 0.0001))
dropout = float(os.getenv('DROPOUT_RATE', 0.25))

mode_str = os.getenv('RUN_MODE', 'train')
mode = tf.estimator.ModeKeys.TRAIN if mode_str == 'train' else tf.estimator.ModeKeys.EVAL

# logging
logger = FileHelper.get_file_console_logger(full_output_name, output_folder, "train.log", True)

# Load data
dataset_length = 0

if mode == tf.estimator.ModeKeys.TRAIN:
    test_messages, test_scores = PreprocessHelper.get_encoded_messages(
        "{}/{}/test/Beauty_test_13862.json.pickle".format(data_path, encoding_name))

    if use_whole_dataset:
        train_messages, train_scores = PreprocessHelper.get_encoded_messages_from_folder(
            "{}/{}/train".format(data_path, encoding_name))
    else:
        train_messages, train_scores = PreprocessHelper.get_encoded_messages(
            "{}/{}/train/Beauty_train_32343.json.pickle".format(data_path, encoding_name))

    dataset_length = len(train_messages)

if mode == tf.estimator.ModeKeys.EVAL:
    if use_whole_dataset:
        test_messages, test_scores = PreprocessHelper.get_encoded_messages_from_folder(
            "{}/{}/test".format(data_path, encoding_name))
    else:
        test_messages, test_scores = PreprocessHelper.get_encoded_messages(
            "{}/{}/test/Beauty_test_13862.json.pickle".format(data_path, encoding_name))

    dataset_length = len(test_messages)

# Initialize graph
tf.reset_default_graph()
tf.set_random_seed(111)

# Initialize inputs
eval_errors = {"sum": 0, "count": 0}
is_train = tf.Variable(True)

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
# flat_layer = cnn.make_flat(pool2_layer)
flat_layer = cnn.make_flat(pool3_layer)
dropout_layer1 = tf.layers.dropout(flat_layer, rate=dropout, training=is_train)
full_connected1 = cnn.full_connection(dropout_layer1, count_neurons=500, name="1")
dropout_layer2 = tf.layers.dropout(full_connected1, rate=dropout, training=is_train)
full_connected2 = cnn.full_connection(dropout_layer2, count_neurons=100, name="2")
dropout_layer3 = tf.layers.dropout(full_connected2, rate=dropout, training=is_train)
full_connected3 = cnn.full_connection(dropout_layer3, count_neurons=1, name="3")

if VERBOSE:
    full_connected3 = tf.Print(full_connected3,
                               [full_connected1, full_connected2, full_connected3],
                               message="This is full_connected 1,2,3: ")

# Optimization
diff = tf.subtract(full_connected3, y_batch)
error = tf.square(diff)
error_sum = tf.reduce_sum(error)

mean_square_error = tf.reduce_mean(error)
root_mean_square_error = tf.sqrt(mean_square_error)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

# Run
saver = tf.train.Saver()

config = tf.ConfigProto(log_device_placement=VERBOSE)
config.gpu_options.allow_growth = True

runner = CNNRunner(VERBOSE, batch_size, logger)

with tf.Session(config=config) as sess:
    def run_train(start_index, end_index, epoch_num):
        is_train.load(True, sess)
        sess.run(optimiser, feed_dict={x_batch: train_messages[start_index:end_index],
                                       y: train_scores[start_index:end_index]})

        is_train.load(False, sess)
        batch_rmse = sess.run(root_mean_square_error, feed_dict={x_batch: train_messages[start_index:end_index],
                                                                 y: train_scores[start_index:end_index]})
        logger.info("Train Epoch {}, Batch {} - {}: RMSE = {}".format(epoch_num, start_index, end_index, batch_rmse))

    def run_eval(start_index, end_index, epoch_num):
        is_train.load(False, sess)
        squared_error_sum = sess.run(error_sum, feed_dict={x_batch: test_messages[start_index:end_index],
                                                           y: test_scores[start_index:end_index]})
        count = end_index - start_index

        eval_errors["sum"] = eval_errors["sum"] + squared_error_sum
        eval_errors["count"] = eval_errors["count"] + count

        batch_rmse = math.sqrt(squared_error_sum/count)

        logger.info("Eval Epoch {}, Batch {} - {}: RMSE = {}".format(epoch_num, start_index, end_index, batch_rmse))


    writer = tf.summary.FileWriter('{}/graph/{}/'.format(output_folder, full_output_name), sess.graph)
    sess.run(tf.global_variables_initializer())

    if mode == tf.estimator.ModeKeys.TRAIN:
        checkpoints_dir = "{}/checkpoints/{}".format(output_folder, full_output_name)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        for epoch in range(epochs):
            runner.call_for_each_batch(dataset_length, epoch, run_train)
            saver.save(sess, "{}/model_epoch{}.ckpt".format(checkpoints_dir, epoch))
            run_eval(0, batch_size, epoch)

    if mode == tf.estimator.ModeKeys.EVAL:
        runner.call_for_each_batch(dataset_length, 0, run_eval)
        total_rmse = math.sqrt(eval_errors["sum"] / eval_errors["count"])
        logger.info("Eval Total RMSE = {}".format(total_rmse))

    writer.close()
