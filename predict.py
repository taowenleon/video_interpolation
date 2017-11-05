import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy
import cv2
import matplotlib as plt

x = tf.placeholder(tf.float32, [1, 320, 240, 6])
y = tf.placeholder(tf.float32, [1, 320, 240, 6])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, )