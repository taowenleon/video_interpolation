# video_interpolation
video frame interpolation with deep learning

This is just a toy model, which following https://arxiv.org/abs/1706.01159.

File lists:
 - generate_dataset.py : converting UCF-101 dataset to tfrecords.
 - train.py : some train op.
 - predict.py : using the trained model to predict the interpolated frames.
 - metrics.py: include PSNR, SSIM and MS-SSIM to evaluate the quality of synthetic frame.
 - ckpt_backup/* : some tained models.
 - utils/* : some scripts may be useful.
 