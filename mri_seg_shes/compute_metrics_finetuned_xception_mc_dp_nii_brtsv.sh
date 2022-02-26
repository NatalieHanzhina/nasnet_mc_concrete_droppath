#!/usr/bin/env bash

########## Xception FPN #############
#python test.py \
python  compute_metrics_on_nn_preds.py \
--num_workers 8 \
--channels 4 \
--pretrained_weights None \
--preprocessing_function caffe \
--network xception_fpn_mc_dp \
--dropout_rate 0.3 \
--batch_size 1 \
--out_channels 2 \
--test_images_dir /media/disk1/mkashirin/mri_data/Brtsv_exp_test/images \
--test_masks_dir /media/disk1/mkashirin/mri_data/Brtsv_exp_test/masks \
--models best_nii_nrh_ft_aug_xception_fpn_mc_dp.h5
