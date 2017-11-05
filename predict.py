import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib as plt

model_path = " "

# x = tf.placeholder(tf.float32, [1, 320, 240, 6])
# y = tf.placeholder(tf.float32, [1, 320, 240, 6])

def read_test_video(path):
    for root, dirs, files in os.walk(path):
        print 'Total %s vidoes!' % str(len(files))
        for video in files:
            capture = cv2.VideoCapture(os.path.join(root, video))
            video_name = str(os.path.splitext(video)[0])
            count = 1
            if capture.isOpened():
                statu, frame = capture.read()
                while statu:
                    frame = tf.image.resize_image_with_crop_or_pad(image=frame, target_height=128, target_width=128)
                    cv2.imwrite('test_frames/' + video_name + os.sep + str(count) + '.png', frame)
                    # frames.append(np.array(frame/255., dtype=np.float32))
                    count = count + 1
                    statu, frame = capture.read()



saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    loss, reconstruct = sess.run([loss_l2, conv_9], feed_dict={inputs:x, labels:y})
    plt.plot(221)
    plt.imshow(reconstruct.eval())
