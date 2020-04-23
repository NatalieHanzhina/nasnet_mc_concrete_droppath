#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--gpu 0 \
--channels 4 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 2 \
--out_channels 2 \
--models_dir nn_models \
--images_dir /media/disk1/mkashirin/data_nii_test/images \
--masks_dir /media/disk1/mkashirin/data_nii_test/masks \
--models best_xception_fpn.h5

