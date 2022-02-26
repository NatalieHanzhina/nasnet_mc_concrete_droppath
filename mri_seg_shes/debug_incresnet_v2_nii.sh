#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 1  \
--network resnetv2 \
--alias nii_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 12  \
--steps_per_epoch 200 \
--epochs 4 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
--log_dir incresnet_v2_nii


#--images_dir /media/disk1/mkashirin/mri_data/data_nii/images \
#--masks_dir /media/disk1/mkashirin/mri_data/data_nii/masks \

#--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
#--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
