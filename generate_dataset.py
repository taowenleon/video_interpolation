import os
import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

RANDOM_SEED = 4242
TFRECIRD_MAX_LEN = 50

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tfrecord_name(outputdir, name, fidx):

    return "%s/%s_%03d.tfrecord" % (outputdir, name, fidx)


def get_dataset_filename(dataset_path, shuffling = True):
    files_list = []

    for root, dirs, files in os.walk(dataset_path):
        for fi in files:
            files_list.append(root.split('/')[-1]+os.sep+fi)

    files_list = sorted(files_list)

    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(files_list)

    return files_list

def split_dataset(files_list):
    total_len = len(files_list)
    train_list = files_list[:int(0.7*total_len)]
    val_list = files_list[int(0.7*total_len)+1:int(0.9*total_len)]
    test_list = files_list[int(0.9*total_len)+1:]

    return train_list, val_list, test_list

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
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'frames': _bytes_feature(frame_input_raw),
            'labels': _bytes_feature(frame_lable_raw)
        }))

        writer.write(example.SerializeToString())

# def crop_frames(frames):

def decoder_tfRecords(tfrecord_files):

    files = tf.gfile.Glob(tfrecord_files)
    file_name_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_name_queue)

    features = tf.parse_single_example(serialized_example, features={
        'frames': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64)
    })

    frames_input = tf.decode_raw(features['frames'], tf.uint8)
    label = tf.decode_raw(features['labels'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    frames_input = tf.reshape(frames_input, [240, 320, 6])
    frames_input = tf.cast(frames_input, tf.float32)/255.
    label = tf.reshape(label, [240, 320, 3])
    label = tf.cast(label, tf.float32)/255.

    return frames_input, label


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
                frames.append(np.array(frame, dtype=np.uint8))
                frame_count = frame_count + 1
            else:
                break
            status, frame = capture.read()
    else:
        print("Open Video Failed!\n")

    capture.release()
    return frames

def video_decoder(dataset_path, files_list, output_path, name = "train", down_sampling = False):
    count = 1
    j = 0
    fidx = 0

    print "Total %d videos!" % len(files_list)

    tf_filename = get_tfrecord_name(output_path, name, fidx)
    writer = tf.python_io.TFRecordWriter(tf_filename)
    for video in files_list:

        if j < TFRECIRD_MAX_LEN:

            '''
            Read Video and extract frames
            '''
            frames = read_video(dataset_path+video)

            if down_sampling:
                frames = frames[::2, :, :, :]

            '''
            Write frames to tfRecords
            '''
            if len(frames) > 0:
                write_tfRecords(writer, frames)
            j += 1

        if j == TFRECIRD_MAX_LEN-1:
            fidx += 1
            j = 0
            writer.close()
            tf_filename = get_tfrecord_name(output_path, name, fidx)
            writer = tf.python_io.TFRecordWriter(tf_filename)

        count += 1
        if count % 10 == 0:
            print "Processing %d videos!" % count
    writer.close()

if __name__ == "__main__":

    dataset_dir = "../../Data/UCF-101/"
    train_output_path = "../../Data/ucf-train"
    val_output_path = "../../Data/ucf-val"
    test_output_path = "../../Data/ucf-test"

    if not tf.gfile.Exists(train_output_path):
        tf.gfile.MakeDirs(train_output_path)
    if not tf.gfile.Exists(val_output_path):
        tf.gfile.MakeDirs(val_output_path)
    if not tf.gfile.Exists(test_output_path):
        tf.gfile.MakeDirs(test_output_path)

    files_list = get_dataset_filename(dataset_dir, shuffling=True)
    # train:val:test == 7 : 2 : 1
    train_list, val_list, test_list = split_dataset(files_list)
    # Generate training tfrecords
    video_decoder(dataset_path=dataset_dir, files_list=train_list, output_path=train_output_path, name="train")
    # Generate validation tfrecords
    video_decoder(dataset_path=dataset_dir, files_list=val_list, output_path=val_output_path, name="val")
    # Generate testing tfrecords
    video_decoder(dataset_path=dataset_dir, files_list=test_list, output_path=test_output_path, name="test")