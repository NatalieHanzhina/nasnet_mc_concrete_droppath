#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--num_workers 8  \
--network xception_fpn_mc \
--freeze_till_layer input_1  \
--alias mc_ \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 7  \
--steps_per_epoch 500 \
--epochs 300 \
--preprocessing_function caffe \
--log_dir xception_fpn_mc
