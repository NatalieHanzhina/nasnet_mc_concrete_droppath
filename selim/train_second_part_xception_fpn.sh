#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

python train.py \
--gpu "0,1,2,3" \
--num_workers 8 \
--network xception_fpn \
--freeze_till_layer input_1 \
--loss double_head_loss \
--optimizer adam \
--learning_rate 0.0001 \
--decay 0.0001 \
--batch_size 16 \
--crop_size 256 \
--steps_per_epoch 500 \
--epochs 98 \
--preprocessing_function caffe \
--weights "nn_models/best_xception_fpn.h5"
