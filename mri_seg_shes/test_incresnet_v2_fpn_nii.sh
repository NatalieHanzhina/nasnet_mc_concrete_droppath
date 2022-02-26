#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--num_workers 8 \
--channels 4 \
--pretrained_weights None \
--preprocessing_function caffe \
--network resnetv2 \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/images \
--test_masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/masks \
--models best_no_pretrain_nii_resnetv2.h5


#--models best_nii_resnetv2.h5
