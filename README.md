# video_interpolation
video frame interpolation with deep learning

This is just a toy model, which following https://arxiv.org/abs/1706.01159.

---

### Requirements:

1. Tensorflow
2. OpenCV 3.0 and above (**Recommended**)

---

### File lists:
 - *generate_dataset.py* : converting UCF-101 dataset to tfrecords.
 - *train.py* : some train op.
 - *predict.py* : using the trained model to predict the interpolated frames.
 - *metrics.py* : including PSNR, SSIM and MS-SSIM to evaluate the quality of synthetic frame.
 - *ckpt_backup/\** : some trained models.
 - *utils/\** : some scripts may be useful.

---

### Results:

![results](https://github.com/taowenleon/video_interpolation/blob/master/results/Figure_1.png)

---

### Usage:

1. Generate the tfrecords to train the model.(You can also use other input format instead of tfrecord. Please refer to:https://www.tensorflow.org/api_guides/python/reading_data)
 ```bash
 python generate_dataset.py
 ```
 This script will generate many *tfrecords* format file, including *train_\*.tfrecord, val_\*.tfrecord, test_\*.tfrecord*, you can modify the parameters to what you want.

2. Train your model after obtained the training tfrecords:
 ```bash
 python train.py
 ```
 This script defines some train ops and hyper-parameters, such as optimization method, learning rate and so on.
 You can use Tensorboard to monitor the train process, using the following command:
 ```bash
 cd log_dir
 tensorboard --logdir="./" --port 6006
 ```
 You can see the training process in http://localhost:6006.

3. Evaluate the trained model:
 ```bash
 python predict.py
 ```
---

### Details:
1. We use UCF-101 for training and testing, you can use any dataset you like to train the model.
2. You can design more elegant or complex model to obtain better results, just rewrite the *model.py*.
3. We will update the repository from time to time, and hope you can join us.
4. If you have any questions, please contact me: taowen0828@gmail.com

---