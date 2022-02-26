#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 8  \
--network resnet152_2 \
--alias nii_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 10  \
--steps_per_epoch 500 \
--epochs 400 \
--preprocessing_function caffe \
--images_dir /media/disk1/mkashirin/mri_data/data_nii/images \
--masks_dir /media/disk1/mkashirin/mri_data/data_nii/masks \
--log_dir resnet152_2_nii
