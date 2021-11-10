#!/usr/bin/env bash

##################### Xception FPN with Sigmoid activation ##############################

train.py \
#--channels 3 \
#--pretrained_weights None \ #imagenet \
#--num_workers 8  \
#--network nasnet_cls \
#--freeze_till_layer input_1  \
#--optimizer adam  \
#--learning_rate 0.0001  \
#--decay 100  \
#--batch_size 32 \
#--steps_per_epoch 500 \
#--epochs 100 \
#--augmentation True \
#--dir "~/cropped_pollen_bayesian"