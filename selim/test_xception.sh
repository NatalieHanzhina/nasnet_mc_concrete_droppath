#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--channels 3 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn \
--batch_size 16 \
--out_channels 2 \
--test_images_dir /media/disk1/mkashirin/data_nii_test/images \
--test_masks_dir /media/disk1/mkashirin/data_nii_test/masks \
--models best_xception_fpn.h5

