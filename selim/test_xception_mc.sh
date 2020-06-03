#!/usr/bin/env bash

########## Xception FPN #############
python test_with_dp.py \
--num_workers 8 \
--channels 3 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn_mc \
--batch_size 1 \
--out_channels 2 \
--models best_dh_dicexception_fpn.h5
