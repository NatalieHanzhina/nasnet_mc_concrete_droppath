#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--channels 4 \
--pretrained_weights None \
--num_workers 8  \
--network xception_fpn \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 16  \
--crop_size 256 \
--steps_per_epoch 500 \
--epochs 2 \
--preprocessing_function caffe \
--images_dir ../data_nii_gz/images \
--masks_dir ../data_nii_gz/masks

python train.py \
--channels 4 \
--pretrained_weights None \
--num_workers 8 \
--network xception_fpn \
--loss double_head_loss \
--optimizer adam \
--learning_rate 0.0001 \
--decay 0.0001 \
--batch_size 16 \
--crop_size 256 \
--steps_per_epoch 500 \
--epochs 98 \
--preprocessing_function caffe \
--images_dir ../data_nii_gz/images \
--masks_dir ../data_nii_gz/masks \
--weights "nn_models/best_xception_fpn.h5"
