import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
# import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from model import net

model_path = "./ckpt/"
frame_path = "./test_video_frames/v_PullUps_g06_c01/"
# x = tf.placeholder(tf.float32, [1, 320, 240, 6])
# y = tf.placeholder(tf.float32, [1, 320, 240, 6])

frame1 = cv2.imread(frame_path+"1.png")
frame2 = cv2.imread(frame_path+"2.png")
frame3 = cv2.imread(frame_path+"3.png")

input_test = np.concatenate((frame1,frame3),axis=2)
input_test = np.array(input_test/255., dtype=np.float32)
input_test = input_test[np.newaxis,:,:,:]
label = frame2
label = np.array(label/255., dtype=np.float32)
label = label[np.newaxis,:,:,:]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + 'trained_ckpt_model-2500.meta')
    # model = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    graph = tf.get_default_graph()
    print graph
    inputs = graph.get_tensor_by_name('inputs:0')
    labels = graph.get_tensor_by_name('labels:0')

    prediction = graph.get_tensor_by_name('Prediction:0')
    print prediction
    # init_op = sess.graph.get_operation_by_name('init')

    # sess.run(init_op)

    prediction= sess.run(prediction, feed_dict={inputs:input_test, labels:label})
    prediction = np.squeeze(prediction)
    frame2_float = frame2/255.
    loss = prediction - frame2_float
    print loss
    # cv2.imshow("prediction", prediction)
    plt.subplot(221)
    plt.imshow(prediction)
    # plt.show()
    plt.subplot(222)
    plt.imshow(frame2)
    plt.show()
    # print "loss = %s, " %loss