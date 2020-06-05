#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--num_workers 8 \
--channels 3 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 3 \
--out_channels 2 \
--models 'best_xception_fpn_ideal[3ch].h5'

