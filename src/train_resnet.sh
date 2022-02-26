#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--num_workers 8  \
--network resnet152_2 \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--log_dir resnet152_2 \
--batch_size 12  \
--steps_per_epoch 500 \
--epochs 2 \
--preprocessing_function caffe

python train.py \
--num_workers 8 \
--network resnet152_2 \
--freeze_till_layer input_1 \
--loss double_head_loss \
--optimizer adam \
--learning_rate 0.0001 \
--decay 0.0001 \
--log_dir resnet152_2 \
--batch_size 12 \
--steps_per_epoch 500 \
--epochs 98 \
--preprocessing_function caffe \
--weights "nn_models/best_resnet152_2.h5"
