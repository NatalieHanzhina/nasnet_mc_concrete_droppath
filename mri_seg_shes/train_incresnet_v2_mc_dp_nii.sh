#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 8  \
--network resnetv2_mc_dp \
--alias nii_no_pretrain_mc_dp=0.3_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--pretrained_weights None  \
--dropout_rate 0.3 \
--decay 0.0001  \
--batch_size 10  \
--steps_per_epoch 400 \
--epochs 400 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
--log_dir no_pretrain_incresnet_v2_mc_dp=0.3_nii
