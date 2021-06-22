#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--pretrained_weights imagenet \
--num_workers 8  \
--network xception_fpn \
--alias nii_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 14 \
--steps_per_epoch 500 \
--epochs 100 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks \
--log_dir xception_fpn_nii
