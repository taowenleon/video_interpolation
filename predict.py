import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import net
from metrics import PSNR, SSIM, MSSSIM

model_path = "./ckpt_queue/"
frame_path = "./test_video_frames/v_BalanceBeam_g05_c01/"
# x = tf.placeholder(tf.float32, [1, 320, 240, 6])

frame1 = cv2.imread(frame_path+"7.png")
frame2 = cv2.imread(frame_path+"8.png")
frame3 = cv2.imread(frame_path+"9.png")

input_test = np.concatenate((frame1, frame3), axis=2)
input_test = np.array(input_test, dtype=np.float32)
input_test = input_test[np.newaxis, :, :, :]/255.
label = frame2
label = np.array(label, dtype=np.float32)
label = label[np.newaxis, :, :, :]/255.

prediction = net(input_test,1,240,320)

inputs = tf.placeholder(tf.float32, shape=[1,240,320,6])

saver = tf.train.Saver()
with tf.Session() as sess:

    model = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, model.model_checkpoint_path)

    graph = tf.get_default_graph()

    prediction= sess.run([prediction], feed_dict={inputs: input_test})
    prediction = np.uint8(np.squeeze(prediction)*255)

    avg_frame = np.uint8((frame1/2.+frame3/2.))

    loss = sum(sum(sum((prediction/255.-frame2/255.) ** 2))) / 2

    psnr = PSNR(prediction, frame2)
    avg_psnr = PSNR(avg_frame, frame2)
    pre_gray = prediction[:,:,0]
    frame2_gray = frame2[:,:,0]
    avg_frame_gray = avg_frame[:,:,0]
    ssim = SSIM(pre_gray, frame2_gray).mean()
    ms_ssim = MSSSIM(pre_gray, frame2_gray)
    avg_SSIM = SSIM(avg_frame_gray, frame2_gray).mean()
    avg_MS_SSIM = MSSSIM(avg_frame_gray, frame2_gray)

    print "Loss = %.2f" % loss
    print "avg_PSNR = %.2f, avg_SSIM = %.4f, avg_MS-SSIM = %.4f" % ( avg_psnr, avg_SSIM, avg_MS_SSIM)
    print "PSNR = %.2f, SSIM = %.4f, MS-SSIM = %.4f" % ( psnr, ssim, ms_ssim)

    # print "loss = %f, PSNR = %f" % (loss, psnr)

    plt.subplot(231)
    plt.imshow(frame1)
    plt.subplot(232)
    plt.imshow(frame3)

    plt.subplot(233)
    plt.imshow(prediction)

    plt.subplot(234)
    plt.imshow(avg_frame)

    plt.subplot(235)
    plt.imshow(frame2)
    plt.show()
