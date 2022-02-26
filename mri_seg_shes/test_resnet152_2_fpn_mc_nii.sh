#!/usr/bin/env bash

########## Xception FPN #############
python test_with_dp.py \
--channels 4 \
--num_workers 8 \
--network resnet152_2_mc \
--pretrained_weights None \
--preprocessing_function caffe \
--dropout_rate 0.3 \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /media/disk1/mkashirin/mri_data/data_nii_test/images \
--test_masks_dir /media/disk1/mkashirin/mri_data/data_nii_test/masks \
--models best_nii_mc_do=0.3_resnet152_2_mc.h5
