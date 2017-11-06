import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib as plt
from model import net

model_path = "./checkpoints/"
frame_path = "./test_video_frames/v_PullUps_g06_c01/"
# x = tf.placeholder(tf.float32, [1, 320, 240, 6])
# y = tf.placeholder(tf.float32, [1, 320, 240, 6])

frame1 = cv2.imread(frame_path+"1.png")
frame2 = cv2.imread(frame_path+"2.png")
frame3 = cv2.imread(frame_path+"3.png")

input = np.concatenate((frame1,frame3),axis=2)
label = frame2

saver = tf.train.import_meta_graph(model_path+'')


with tf.Session() as sess:
    model = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, model.model_checkpoint_path)
    _, loss, prediction= sess.run([loss_l2, prediction], feed_dict={input:input, label:label})
    plt.plot(221)
    plt.imshow(prediction.eval())
    print "loss = %s, " %loss