# video_interpolation
video frame interpolation with deep learning

This is just a toy model, which following https://arxiv.org/abs/1706.01159.

### Requirement:

1. Tensorflow
2. OpcnCV 3.0 (ReCommend)

### File lists:
 - *generate_dataset.py* : converting UCF-101 dataset to tfrecords.
 - *train.py* : some train op.
 - *predict.py* : using the trained model to predict the interpolated frames.
 - *metrics.py* : include PSNR, SSIM and MS-SSIM to evaluate the quality of synthetic frame.
 - *ckpt_backup/\** : some tained models.
 - *utils/\** : some scripts may be useful.

### Results:

![](https://github.com/taowenleon/video_interpolation/blob/master/results/Figure_1.png)

### Details:
1. We use UCF-101 for training and testing, you can use any dataset you like to train the model.
2. You can design more elegant or complex model to obtain better results, just rewrite the *model.py*.
3. We will update the repository from time to time, and hope you can join us.
4. if you have any questions, please contact me: taowen0828@gmail.com