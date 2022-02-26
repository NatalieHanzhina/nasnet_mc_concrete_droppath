#!/usr/bin/env bash

########## Xception FPN #############
python test_with_dp.py \
--channels 4 \
--num_workers 8 \
--network resnetv2_sch_dp \
--pretrained_weights None \
--preprocessing_function caffe \
--dropout_rate 0.3 \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/images \
--test_masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/masks \
--models best_nii_no_pretrain_mc_sch_dp=0.3_resnetv2_sch_dp.h5
