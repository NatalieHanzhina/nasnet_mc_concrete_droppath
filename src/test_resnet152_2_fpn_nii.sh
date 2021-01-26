#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--num_workers 8 \
--channels 4 \
--pretrained_weights None \
--preprocessing_function caffe \
--network resnet152_2 \
--batch_size 3 \
--out_channels 2 \
--test_images_dir /media/disk1/mkashirin/mri_data/data_nii_test/images \
--test_masks_dir /media/disk1/mkashirin/mri_data/data_nii_test/masks \
--models best_nii_resnet152_2.h5
