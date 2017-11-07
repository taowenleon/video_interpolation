import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just show warnings and errors
import numpy as np
import tensorflow as tf
from model import net

# import tensorflow.contrib.slim as slim

import sys
sys.path.append("../../")

from dataset2 import decoder_tfRecords

# checkpoint_directory = './checkpoints/'
checkpoint_directory = './ckpt/'
log_directory = '../../log/video_interpolation/'
tfrecords_path = '../../Data/UCF101_dataset_float32.tfrecords'

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

BATCH_SIZE = 32
START_LEARNING_RATE = 0.001
MAX_EPOCHES = 100000
height = 128
width = 128
depth = 3

frames, ground_truth = decoder_tfRecords(tfrecords_path)

img_batch, label_batch = tf.train.shuffle_batch(
    [frames, ground_truth],
    batch_size=8,
    capacity=1000,
    num_threads=4,
    min_after_dequeue=8
)

inputs = tf.placeholder(tf.float32, [None, height, width, depth*2], name='inputs')
labels = tf.placeholder(tf.float32, [None, height, width, depth], name='labels')

global_steps = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_steps')
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_steps, 5000, 0.96, staircase=True)

_, loss_l2 = net(inputs, labels)

loss_l2 = loss_l2/(BATCH_SIZE*height*width*depth)

tf.summary.scalar('loss', loss_l2)
tf.summary.scalar('leaning rate', learning_rate)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_l2, global_step=global_steps)

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    writer = tf.summary.FileWriter(log_directory, sess.graph)

    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)

    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    start = global_steps.eval()
    print "Start from:", start

    for i in range(start, MAX_EPOCHES):
        _, loss, summary = sess.run([optimizer, loss_l2, merged], feed_dict={inputs:img_batch.eval(), labels:label_batch.eval()})
        writer.add_summary(summary, i)

        global_steps.assign(i).eval()
        if i % 100 == 0:
            saver.save(sess, checkpoint_directory+'trained_ckpt_model',
                       global_step=global_steps)

        print 'Iteration = %s, loss = %s' % (str(i), str(loss))
writer.close()