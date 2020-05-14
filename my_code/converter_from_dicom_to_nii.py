import os
from argparse import ArgumentParser

import pydicom
import shutil
import numpy as np
import nibabel as nib


IMAGES_MODES = ['ML_DWI', 'ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
DIR_FILTER_PREFIX = 'Ax'


def main():
    data_dir, save_path = get_args()
    files_to_parse_gen = get_files_to_parse_generator(data_dir)
    open_and_write_files_to_parse(files_to_parse_gen, data_dir, save_path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_data_dir', help='Path to data dir to parse')
    parser.add_argument('--save_path', help='Path to data dir to save results', default='converted_nii_gz')
    args = parser.parse_args()
    return args.path_to_data_dir, args.save_path


def get_files_to_parse_generator(dir_path):
    dir_gen = os.walk(dir_path)
    for cur_dir, sub_dirs, _ in dir_gen:
        correct_modes = [subdir for subdir in sub_dirs if subdir in IMAGES_MODES]
        if len(correct_modes) > 0:
            yield cur_dir, correct_modes


def open_and_write_files_to_parse(files_gen, source_dir, save_dir_path):
    #source_dir_name = os.path.split(source_dir)[1]
    for parent_path, channels in files_gen:
        save_msk_path = str.replace(parent_path, source_dir[:-1], os.path.join(save_dir_path, 'masks'))
        save_msk_path = os.path.join(save_msk_path, 'seg')
        print(parent_path)
        print(channels)
        print(save_msk_path)
        min_img_count = 1e10
        masks = []

        for channel in channels:
            channel_path = os.path.join(parent_path, channel)
            if source_dir[-1] == '/':
                save_img_channel_path = str.replace(channel_path, source_dir[:-1], os.path.join(save_dir_path, 'images'))
            else:
                save_img_channel_path = str.replace(channel_path, source_dir, os.path.join(save_dir_path, 'images'))

            print(save_img_channel_path)
            #input()

            channel_subdirs_names = [k for k in os.listdir(channel_path) if os.path.isdir(os.path.join(channel_path, k))]
            if len(channel_subdirs_names) > 1:
                filtered_channel_subdirs_names = [d for d in channel_subdirs_names if DIR_FILTER_PREFIX in d]
            else:
                filtered_channel_subdirs_names = channel_subdirs_names

            assert len(filtered_channel_subdirs_names) == 1
            ch_sbdir = filtered_channel_subdirs_names[0]
            ch_sbdir_path = os.path.join(channel_path, ch_sbdir)


            files_names = os.listdir(ch_sbdir_path)

            if len(files_names) > 0 and check_files_format(files_names, ['dcm', 'dcm.lbl']):
                img_files = [pydicom.dcmread(os.path.join(ch_sbdir_path, f_n)).pixel_array for f_n in files_names if f_n.endswith('.dcm')]
                min_img_count = min(min_img_count, len(img_files))
                mask_files = [pydicom.dcmread(os.path.join(ch_sbdir_path, f_n), force=True).pixel_array for f_n in files_names if f_n.endswith('.dcm.lbl')]

                all_files = [f_n for f_n in files_names if f_n.endswith('.dcm') or f_n.endswith('.dcm.lbl')]
                for f in all_files:
                    print(f)
                print(os.path.split(save_img_channel_path)[0])
                shutil.rmtree(os.path.split(save_img_channel_path)[0], ignore_errors=True)
                os.makedirs(os.path.split(save_img_channel_path)[0])
                input()
                nib_imgs = nib.Nifti1Image(np.asarray(img_files), np.eye(4))
                nib.save(nib_imgs, save_img_channel_path+'.nii')


def check_files_format(file_names_to_ckeck, formats=['dcm']):
    formats = list(formats)
    for f in file_names_to_ckeck:
        formats_check = False
        for format in formats:
            if f.endswith(format if '.' in format else'.' + format):
                formats_check = True
        if not formats_check:
            return False
    return True


if __name__ == '__main__':
    main()
