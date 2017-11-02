import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just show warnings and errors
import numpy as np
import tensorflow as tf
from model import net

import tensorflow.contrib.slim as slim

import sys
sys.path.append("../../")

from dataset2 import decoder_tfRecords

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MAX_EPOCHES = 100000000
height = 128
width = 128
depth = 3


frames, ground_truth = decoder_tfRecords('../../Data/UCF101_dataset_float32.tfrecords')

# frames = tf.image.resize_image_with_crop_or_pad(image=frames, target_height=height, target_width=width)
# ground_truth = tf.image.resize_image_with_crop_or_pad(image=ground_truth, target_height=height, target_width=width)

img_batch, label_batch = tf.train.shuffle_batch(
    [frames, ground_truth],
    batch_size=8,
    capacity=1000,
    num_threads=4,
    min_after_dequeue=8
)

inputs = tf.placeholder(tf.float32, [None, height, width, depth*2])
labels = tf.placeholder(tf.float32, [None, height, width, depth])

loss_l2 = net(inputs, labels)
tf.summary.scalar('loss', loss_l2)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_l2)

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    writer = tf.summary.FileWriter('./log',sess.graph)
    for i in range(MAX_EPOCHES):
        _, loss, summary = sess.run([optimizer, loss_l2, merged], feed_dict={inputs:img_batch.eval(), labels:label_batch.eval()})
        writer.add_summary(summary, i)
        print 'Iteration = %s, loss = %s' % (str(i), str(loss/BATCH_SIZE))
writer.close()