#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--num_workers 8  \
--network nasnet_sch_dp \
--alias nii_mc_sch_dp=0.5_ \
--resize_size 256  \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--dropout_rate 0.2538 \
--decay 0.0001  \
--batch_size 10  \
--steps_per_epoch 500 \
--epochs 400 \
--preprocessing_function caffe \
--images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/images  \
--masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii/masks  \
--log_dir nasnet_mc_sch_dp=0.5_nii
