#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--pretrained_weights imagenet \
--num_workers 8  \
--network xception_fpn \
--alias nii_nrh_ft_ \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 12 \
--resize_size 256 \
--steps_per_epoch 500 \
--epochs 20 \
--preprocessing_function caffe \
--images_dir /media/disk1/mkashirin/Brtsv_exp/images \
--masks_dir /media/disk1/mkashirin/Brtsv_exp/masks \
--log_dir xception_fpn_nii_nrh_ft  \
--weights "nn_models/best_nii_xception_fpn.h5"

#aggregation
