#!/usr/bin/env bash

########## Xception FPN #############
python test_with_dp.py \
--num_workers 8 \
--channels 4 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn_mc \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /media/disk1/mkashirin/Brtsv_exp_test/images \
--test_masks_dir /media/disk1/mkashirin/Brtsv_exp_test/masks \
--models best_nii_nrh_ft_aug_xception_fpn_mc.h5
