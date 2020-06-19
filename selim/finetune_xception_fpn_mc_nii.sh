#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4  \
--num_workers 8  \
--network xception_fpn_mc \
--alias nii_nrh_ft_aug  \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 12  \
--resize_size 256 \
--steps_per_epoch 500 \
--epochs 20 \
--preprocessing_function caffe  \
--images_dir /media/disk1/mkashirin/Brtsv_exp/images  \
--masks_dir /media/disk1/mkashirin/Brtsv_exp/masks  \
--log_dir xception_fpn_mc_nii_nrh_ft_aug \
--weights "nn_models/best_nii_xception_fpn_mc.h5"

#aggregation
