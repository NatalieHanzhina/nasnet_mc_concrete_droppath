#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--pretrained_weights imagenet \
--num_workers 8 \
--network xception_fpn \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam \
--learning_rate 0.0001 \
--decay 0.0001 \
--batch_size 16 \
--steps_per_epoch 500 \
--epochs 98 \
--preprocessing_function caffe \
--images_dir /media/disk1/mkashirin/data_nii/images \
--masks_dir /media/disk1/mkashirin/data_nii/masks \
--log_dir xception_fpn_nii \
--weights "nn_models/best_xception_fpn.h5"
