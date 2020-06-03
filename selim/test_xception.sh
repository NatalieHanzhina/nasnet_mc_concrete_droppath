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
--models best_dh_dicexception_fpn_locally_trained.h5

