#!/usr/bin/env bash

########## Xception FPN #############

python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 8 \
--out_channels 2 \
--models_dir nn_models \
--models best_dh_dicexception_fpn.h5


python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 8 \
--out_channels 2 \
--models_dir nn_models \
--models best_dh_dicexception_fpn_locally_trained.h5


python test.py \
--gpu 0 \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 8 \
--out_channels 2 \
--models_dir nn_models \
--models best_xception_fpn.h5

