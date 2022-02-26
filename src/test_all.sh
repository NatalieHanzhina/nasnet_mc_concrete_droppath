#!/usr/bin/env bash

########## Resnet101 full masks for extension #############
python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network resnet101_2 \
--batch_size 16 \
--out_channels 2 \
--models_dir nn_models \
--models best_resnet101_2.h5


########## Xception FPN #############
python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 16 \
--out_channels 2 \
--models_dir nn_models \
--models best_xception_fpn.h5


########## Resnet152 sigmoid 2 channels #############
python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network resnet152_2 \
--batch_size 14 \
--out_channels 2 \
--models_dir nn_models \
--models best_resnet152_2.h5
