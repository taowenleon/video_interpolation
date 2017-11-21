import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
# import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from model import net

model_path = "./ckpt_queue/"
frame_path = "./test_video_frames/v_PullUps_g12_c02/"
# x = tf.placeholder(tf.float32, [1, 320, 240, 6])
# y = tf.placeholder(tf.float32, [1, 320, 240, 6])

frame1 = cv2.imread(frame_path+"11.png")
frame2 = cv2.imread(frame_path+"12.png")
frame3 = cv2.imread(frame_path+"13.png")

input_test = np.concatenate((frame1, frame3), axis=2)
input_test = np.array(input_test, dtype=np.float32)
input_test = input_test[np.newaxis, :, :, :]/255.
label = frame2
label = np.array(label, dtype=np.float32)
label = label[np.newaxis, :, :, :]/255.

prediction = net(input_test)

inputs = tf.placeholder(tf.float32, shape=[1,128,128,6])

saver = tf.train.Saver()
with tf.Session() as sess:

    model = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, model.model_checkpoint_path)

    graph = tf.get_default_graph()

    prediction= sess.run([prediction], feed_dict={inputs: input_test})
    prediction = np.squeeze(prediction)

    frame2_float = frame2/255.
    loss = sum(sum(sum((prediction-frame2_float) ** 2))) / 2
    print loss

    plt.subplot(121)
    plt.imshow(prediction)

    plt.subplot(122)
    plt.imshow(frame2)
    plt.show()
