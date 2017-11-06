from __future__ import division
from __future__ import print_function

import tensorflow as tf




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def net(input, label):
    # Conv_1
    W_conv1 = weight_variable([3, 3, 6, 128])
    b_conv1 = bias_variable([128])
    conv_1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)

    # Conv_2
    W_conv2 = weight_variable([3, 3, 128, 256])
    b_conv2 = bias_variable([256])
    conv_2 = tf.nn.relu(conv2d(conv_1, W_conv2) + b_conv2)

    # Conv_3
    W_conv3 = weight_variable([3, 3, 256, 512])
    b_conv3 = bias_variable([512])
    conv_3 = tf.nn.relu(conv2d(conv_2, W_conv3) + b_conv3)
    #conv_3_pooling = max_pooling_2x2(conv_3)

    # Conv_4
    W_conv4 = weight_variable([3, 3, 512, 512])
    b_conv4 = bias_variable([512])
    conv_4 = tf.nn.relu(conv2d(conv_3, W_conv4) + b_conv4)
    # conv_4_pooling = max_pooling_2x2(conv_4)

    # Conv_5
    W_conv5 = weight_variable([3, 3, 512, 1024])
    b_conv5 = bias_variable([1024])
    conv_5 = tf.nn.relu(conv2d(conv_4, W_conv5) + b_conv5)

    # Conv6
    W_conv6 = weight_variable([3, 3, 1024, 512])
    b_conv6 = bias_variable([512])
    conv_6 = tf.nn.relu(conv2d(conv_5, W_conv6) + b_conv6)

    # Conv_7
    W_conv7 = weight_variable([3, 3, 512, 256])
    b_conv7 = bias_variable([256])
    conv_7 = tf.nn.relu(conv2d(conv_6, W_conv7) + b_conv7)

    # Conv_8
    W_conv8 = weight_variable([3, 3, 256, 64])
    b_conv8 = bias_variable([64])
    conv_8 = tf.nn.relu(conv2d(conv_7, W_conv8) + b_conv8)

    # Conv_9
    W_conv9 = weight_variable([3, 3, 64, 3])
    b_conv9 = bias_variable([3])
    conv_9 = conv2d(conv_8, W_conv9) + b_conv9

    loss_l2 = tf.nn.l2_loss(conv_9-label)

    return conv_9, loss_l2