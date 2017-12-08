"""
Author: taowen
Date: 2017-11-15

"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def weight_variable(shape):
    # initial = tf.random_normal_initializer(0, 0.02)
    initial = tf.truncated_normal(shape, stddev=1e-2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, batch_size, out_channels):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    return tf.nn.conv2d_transpose(x, W, [batch_size, in_height*2, in_width*2, out_channels], strides=[1, 2, 2, 1], padding="SAME")

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def net(inputs, batch_size):
    frame1 = inputs[:, :, :, 0:3]
    frame2 = inputs[:, :, :, 3:6]
    height = inputs.get_shape()[1]
    width = inputs.get_shape()[2]

    # Conv1
    with tf.name_scope('Conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 64])
        b_conv1 = bias_variable([64])
        conv1_1 = tf.nn.relu(conv2d(frame1, W_conv1) + b_conv1, name='Conv1_1')
        conv1_2 = tf.nn.relu(conv2d(frame2, W_conv1) + b_conv1, name='Conv1_2')

    # max_pooling1
    with tf.name_scope('Pooling1') as scope:
        pooling1_1 = max_pooling_2x2(conv1_1)
        pooling1_2 = max_pooling_2x2(conv1_2)

    # Conv2
    with tf.name_scope('Conv2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        conv2_1 = tf.nn.relu(conv2d(pooling1_1, W_conv2) + b_conv2, name='Conv2_1')
        conv2_2 = tf.nn.relu(conv2d(pooling1_2, W_conv2) + b_conv2, name='Conv2_2')

    # Conv3
    with tf.name_scope('Conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        conv3_1 = tf.nn.relu(conv2d(conv2_1, W_conv3) + b_conv3, name='Conv3_1')
        conv3_2 = tf.nn.relu(conv2d(conv2_2, W_conv3) + b_conv3, name='Conv3_2')

    # max_pooling2
    with tf.name_scope('Pooling2') as scope:
        pooling2_1 = max_pooling_2x2(conv3_1)
        pooling2_2 = max_pooling_2x2(conv3_2)

    # Conv4
    with tf.name_scope('Conv4') as scope:
        W_conv4 = weight_variable([3, 3, 64, 128])
        b_conv4 = bias_variable([128])
        conv4_1 = tf.nn.relu(conv2d(pooling2_1, W_conv4) + b_conv4, name='Conv4_1')
        conv4_2 = tf.nn.relu(conv2d(pooling2_2, W_conv4) + b_conv4, name='Conv4_2')

    # Conv5
    with tf.name_scope('Conv5') as scope:
        W_conv5 = weight_variable([3, 3, 128, 128])
        b_conv5 = bias_variable([128])
        conv5_1 = tf.nn.relu(conv2d(conv4_1, W_conv5) + b_conv5, name='Conv5_1')
        conv5_2 = tf.nn.relu(conv2d(conv4_2, W_conv5) + b_conv5, name='Conv5_2')

    # max_pooling3
    with tf.name_scope('Pooling3') as scope:
        pooling3_1 = max_pooling_2x2(conv5_1)
        pooling3_2 = max_pooling_2x2(conv5_2)

    with tf.name_scope('Merge_add') as scope:
        merge = tf.add(pooling3_1, pooling3_2, name="Merge")

    with tf.name_scope('DeConv1') as scope:
        W_deconv1 = weight_variable([3, 3, 64, 128])
        deconv1 = deconv2d(merge, W_deconv1, batch_size, 64)

    with tf.name_scope('DeConv2') as scope:
        W_deconv2 = weight_variable([3, 3, 64, 64])
        deconv2 = deconv2d(deconv1, W_deconv2, batch_size, 64)

    with tf.name_scope('DeConv3') as scope:
        W_deconv3 = weight_variable([3, 3, 64, 64])
        deconv3 = deconv2d(deconv2, W_deconv3, batch_size, 64)

    with tf.name_scope('Conv6') as scope:
        W_conv6 = weight_variable([3, 3, 64, 3])
        b_conv6 = bias_variable([3])
        conv6 =tf.nn.sigmoid(conv2d(deconv3, W_conv6) + b_conv6, name='Conv6')

    return conv6