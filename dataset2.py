import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def video_decoder(path):
    count = 1

    tfRecords = '../../Data/UCF101_dataset_train_320_240.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfRecords)
    for root, dirs, files in os.walk(path):
        print 'Total %s vidoes!' % str(len(files))
        for video in files:
            video_path = os.path.join(root, video)

            if count % 10 == 0:
                print "Processing %d videos!" % count

            '''
            Read Video and extract frames
            '''
            frames = read_video(video_path)

            '''
            Write frames to tfRecords
            '''
            if len(frames) > 0:
                write_tfRecords(writer, frames)
            count = count + 1

    writer.close()

def write_tfRecords(writer, frames):
    frames = np.array(frames)
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
        frame_input_raw = train_doublets[i, :, :, :].tobytes()
        frame_lable_raw = train_label[i, :, :, :].tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            # 'height': _int64_feature(height),
            # 'width': _int64_feature(width),
            # 'depth': _int64_feature(depth),
            'frames': _bytes_feature(frame_input_raw),
            'labels': _bytes_feature(frame_lable_raw)
        }))

        writer.write(example.SerializeToString())

# def crop_frames(frames):



def read_video(video_path):
    frames = []
    frame_count = 1
    capture = cv2.VideoCapture(video_path)
    nFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if nFrames <100:
        return frames

    if capture.isOpened():
        status, frame = capture.read()
        while status:
            if frame_count <= 100:
                # frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_CUBIC)
                # frame = Image.fromarray(frame, 'RGB')
                frames.append(np.array(frame, dtype=np.uint8))
                frame_count = frame_count + 1
            else:
                break
            status, frame = capture.read()
    else:
        print("Open Video Failed!\n")

    capture.release()
    return frames

def decoder_tfRecords(file_name):
    file_name_queue = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_name_queue)

    features = tf.parse_single_example(serialized_example, features={
        'frames': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.string),
        # 'height': tf.FixedLenFeature([], tf.int64),
        # 'width': tf.FixedLenFeature([], tf.int64),
        # 'depth': tf.FixedLenFeature([], tf.int64)
    })

    frames_input = tf.decode_raw(features['frames'], tf.uint8)
    label = tf.decode_raw(features['labels'], tf.uint8)
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)

    frames_input = tf.reshape(frames_input, [240, 320, 6])
    frames_input = tf.cast(frames_input, tf.float32)/255.
    label = tf.reshape(label, [240, 320, 3])
    label = tf.cast(label, tf.float32)/255.

    return frames_input, label

if __name__ == "__main__":

    video_decoder("/home/taowen/Workspace/Data/UCF-101/")
