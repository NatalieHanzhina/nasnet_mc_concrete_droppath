#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--pretrained_weights imagenet \
--num_workers 8  \
--network xception_fpn_mc  \
--alias nii_mc_do=0.3_  \
--freeze_till_layer input_1  \
--loss double_head_loss  \
--optimizer adam  \
--learning_rate 0.0001  \
--dropout_rate 0.3  \
--decay 0.0001  \
--batch_size 14  \
--steps_per_epoch 500 \
--epochs 300 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks \
--log_dir xception_fpn_mc_do=0.3_nii
