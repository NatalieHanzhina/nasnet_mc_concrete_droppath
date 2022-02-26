#!/usr/bin/env bash

########## Xception FPN #############
python test.py \
--channels 4 \
--num_workers 8 \
--network nasnet_mc_dp_test \
--dropout_rate 0.0 \
--pretrained_weights None \
--preprocessing_function caffe \
--resize_size 256  \
--times_sample_per_test 1 \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/images \
--test_masks_dir /mnt/tank/scratch/mkashirin/datasets/mri_data/data_nii_test/masks \
--models best_nii_mc_sch_dp=0.3_nasnet_mc_sch_dp.h5

