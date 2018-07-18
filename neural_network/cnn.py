import tensorflow as tf


def cnn_conv_layer(data, name, filter_shape, filters=16, channels=1):
    # setup the filter input shape for tf.nn.conv_2d
    name = "conv_" + name
    conv_filt_shape = [filter_shape[0], filter_shape[1], channels, filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([filters]), name=name + '_b')

    # Convolutional Layer #1
    out_layer = tf.nn.conv2d(data, filter=weights, strides=[1, filter_shape[0], 1, 1],
                             padding='SAME', name=name + "_conv")

    # add the bias
    out_layer += bias
    out_layer = tf.nn.relu(out_layer, name=name + "_relu")
    return out_layer


def max_pooling(in_layer, name, pool_shape=None):
    if pool_shape is None:
        pool_shape = [1, 5]

    pool_name = "max_pool_{}".format(name)
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 2, 1]
    out_layer = tf.nn.max_pool(in_layer, name=pool_name, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def full_connection(in_layer, count_neurons, name):
    name = "fc_" + name

    wd = tf.Variable(tf.truncated_normal([in_layer.shape[1].value, count_neurons], stddev=0.03), name=name + '_wd')
    bd = tf.Variable(tf.truncated_normal([count_neurons], stddev=0.01), name=name + '_bd')
    dense_layer = tf.matmul(in_layer, wd) + bd
    # dense_layer = tf.nn.relu(dense_layer)

    return dense_layer


def make_flat(in_layer):
    return tf.reshape(in_layer, [-1, in_layer.shape[1].value * in_layer.shape[2].value * in_layer.shape[3].value])


class CNNRunner:
    def __init__(self, verbose, batch_size, logger):
        self.VERBOSE = verbose
        self.batch_size = batch_size
        self.logger = logger

    def call_for_each_batch(self, dataset_length, epoch, call_func, messages, scores):
        start_index = 0
        end_index = self.batch_size

        while start_index < dataset_length:
            if dataset_length < end_index:
                end_index = dataset_length

            if self.VERBOSE:
                self.logger.info("Epoch {}, Batch {} - {}: ".format(epoch, start_index, end_index, epoch))

            call_func(start_index, end_index, epoch, messages, scores)

            start_index += self.batch_size
            end_index += self.batch_size