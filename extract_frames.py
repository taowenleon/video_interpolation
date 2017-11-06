import os
import cv2

def read_video(path):
    frames_dir = './test_video_frames/'
    for root, dirs, files in os.walk(path):
        print 'Total %s vidoes!' % str(len(files))
        for video in files:
            capture = cv2.VideoCapture(os.path.join(root, video))
            video_name = str(os.path.splitext(video)[0])
            count = 1
            if not os.path.exists(frames_dir+video_name):
                os.makedirs(frames_dir+video_name)

            if capture.isOpened():
                statu, frame = capture.read()
                while statu:
                    frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(frames_dir + video_name + os.sep + str(count) + '.png', frame)
                    # frames.append(np.array(frame/255., dtype=np.float32))
                    count = count + 1
                    statu, frame = capture.read()

if __name__ == '__main__':
    read_video('../../Data/UCF-101-Test/')