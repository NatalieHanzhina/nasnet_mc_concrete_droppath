#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 1  \
--network resnetv2_mc_dp \
--alias nii_mc_dp=0.3_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--dropout_rate 0.3 \
--decay 0.0001  \
--batch_size 1  \
--steps_per_epoch 4 \
--epochs 2 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
--log_dir incresnet_v2_mc_dp=0.3_nii
