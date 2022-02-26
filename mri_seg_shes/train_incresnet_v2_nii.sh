#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 8  \
--network resnetv2 \
--alias no_pretrain_nii_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--pretrained_weights None  \
--learning_rate 0.0001  \
--dropout_rate 0.3 \
--decay 0.0001  \
--batch_size 10  \
--steps_per_epoch 400 \
--epochs 300 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
--log_dir incresnet_v2_no_pretrain_nii
