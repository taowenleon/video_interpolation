import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def video_decoder(path):
    count = 1
    tfRecords = '../Data/UCF101_dataset_float32.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfRecords)
    for root, dirs, files in os.walk(path):
        print 'Total %s vidoes!' % str(len(files))
        for video in files:
            frames = []
            frame_count = 1
            capture = cv2.VideoCapture(os.path.join(root, video))

            nFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if nFrames > 100:
                continue

            if count % 10 == 0:
                print "Processing %s video......" % str(count)

            if capture.isOpened():
                statu, frame = capture.read()
                while statu:
                    if frame_count <= 100:
                        frame = tf.image.resize_image_with_crop_or_pad(image=frame, target_height=128, target_width=128)
                        frames.append(np.array(frame/255., dtype=np.float32))
                        frame_count = frame_count + 1
                    else:
                        break
                    statu, frame = capture.read()

                frames = np.stack(frames)
                height = frames.shape[1]
                width = frames.shape[2]
                depth = frames.shape[3]

                train_before = frames[:-2, :, :, :]
                train_middle = frames[1:-1, :, :, :]
                train_after = frames[2:, :, :, :]

                train_doublets = np.concatenate((train_before, train_after), axis=3)
                train_label = train_middle

                for i in range(train_label.shape[0]):
                    # plt.subplot(221)
                    # plt.imshow(train_doublets[2,:,:,0:3])
                    # plt.subplot(222)
                    # plt.imshow(train_doublets[2, :, :, 3:6])
                    # plt.subplot(223)
                    # plt.imshow(train_label[2, :, :, :])
                    # plt.show()
                    frame_input_raw = train_doublets[i, :, :, :].tostring()
                    frame_lable_raw = train_label[i, :, :, :].tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'depth': _int64_feature(depth),
                        'frames': _bytes_feature(frame_input_raw),
                        'labels': _bytes_feature(frame_lable_raw)
                    }))

                    writer.write(example.SerializeToString())

                capture.release()

                # return frames
            else:
                print("Open Video Failed!\n")
                continue

            count = count + 1
    writer.close()

def decoder_tfRecords(file_name):
    file_name_queue = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_name_queue)

    features = tf.parse_single_example(serialized_example, features={
        'frames': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64)
    })

    frames_input = tf.decode_raw(features['frames'], tf.float32)
    label = tf.decode_raw(features['labels'], tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    frames_input = tf.reshape(frames_input, [128, 128, 6])
    label = tf.reshape(label, [128, 128, 3])

    return frames_input, label

if __name__ == "__main__":

    video_decoder("/home/taowen/Workspace/Data/UCF-101-Train/")
