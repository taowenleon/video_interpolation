"""
Author: taowen
Date: 2017-11-15

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just show warnings and errors
import numpy as np
import tensorflow as tf
import cv2
# import math
from PIL import Image
from model import net

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

# import tensorflow.contrib.slim as slim

import sys
sys.path.append("../../")

from generate_dataset import decoder_tfRecords

# checkpoint_directory = './checkpoints/'

checkpoint_directory = './ckpt/'
log_directory = '../../log/video_interpolation/model_Y_Net/'
tfrecords_path = '../../Data/ucf-train/train_*.tfrecord'

if not tf.gfile.Exists(checkpoint_directory):
    tf.gfile.MakeDirs(checkpoint_directory)

if not tf.gfile.Exists(log_directory):
    tf.gfile.MakeDirs(log_directory)

BATCH_SIZE = 16
START_LEARNING_RATE = 1e-4
MAX_EPOCHES = 100000
height = 240
width = 320
depth = 3

swd = "./batch_test/"

frames, ground_truth = decoder_tfRecords(tfrecords_path)

img_batch, label_batch = tf.train.shuffle_batch(
    [frames, ground_truth],
    batch_size=BATCH_SIZE,
    capacity=3200,
    num_threads=4,
    min_after_dequeue=1600,
    allow_smaller_final_batch=True
)

global_steps = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_steps')
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_steps, 5000, 0.96, staircase=True)
# _, loss_l2 = net(inputs, labels)
predection = net(img_batch, BATCH_SIZE)

loss_l2 = tf.nn.l2_loss(predection-label_batch)

loss_l2 = loss_l2/BATCH_SIZE

tf.summary.scalar('loss', loss_l2)
tf.summary.scalar('leaning rate', learning_rate)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_l2, global_step=global_steps)

# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    writer = tf.summary.FileWriter(log_directory, sess.graph)

    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    start = global_steps.eval()
    print "Start from:", start

    for i in range(start, MAX_EPOCHES):

        _, _, loss, summary = sess.run([optimizer, predection, loss_l2, merged])
        writer.add_summary(summary, i)

        global_steps.assign(i).eval()
        if i % 500 == 0:

            saver.save(sess, checkpoint_directory+'trained_ckpt_model',
                       global_step=global_steps)

        print 'Iteration = %s, loss = %s' % (str(i), str(loss))
writer.close()


'''
The following codes check the the inputs and labels of the network.
'''
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(sess = sess,coord=coord)
#     diff_sum = 0
#     for i in range(10):
#         example, l = sess.run([img_batch,label_batch])
#         # example = example/255.
#         # l = l/255.
#         for j in range(BATCH_SIZE):
#             # frame1 = Image.fromarray(example[j][:,:,0:3]/255., 'RGB')
#             # frame3 = Image.fromarray(example[j][:,:,3:6]/255., 'RGB')
#             # frame2 = Image.fromarray(l[j]/255., 'RGB')
#             frame1 = example[j][:,:,0:3]
#             frame3 = example[j][:, :, 3:6]
#             diff = np.sum(np.sum(np.sum(abs(frame3*255-frame1*255), axis=2), axis=1),axis=0)
#             diff_sum = diff_sum+diff
#             print diff
#             frame2 = l[j]
#             # sigle_label = l[j]
#
#             cv2.imwrite(swd + 'batch_' + str(i) + '_' + 'size_' + str(j) + '_' + 'frame1' + '.png', frame1*255)
#             cv2.imwrite(swd + 'batch_' + str(i) + '_' + 'size_' + str(j) + '_' + 'frame2' + '.png', frame2*255)
#             cv2.imwrite(swd + 'batch_' + str(i) + '_' + 'size_' + str(j) + '_' + 'frame3' + '.png', frame3*255)
#
#             # print(example, l)
#     print "average diff = %f" % (diff_sum/(10*BATCH_SIZE))
#
#     coord.request_stop()
#     coord.join(threads)

