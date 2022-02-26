import gc
import os
import shutil
from argparse import ArgumentParser

import imageio
import numpy as np
import pydicom
import pydicom.uid

IMAGES_MODES = ['ML_DWI', 'ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
DIR_FILTER_PREFIX = 'Ax'
DIR_FILTER_NAME = 'AX'

EXCLUDED_DIRS = 0


def main():
    dirs = get_args()
    count_dirs_sbdirs(dirs)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_data_dir', help='Path to data dir to parse')
    parser.add_argument('imgs_dir', help='Path to data dir to parse')
    parser.add_argument('msks_dir', help='Path to data dir to parse')
    args = parser.parse_args()
    return [args.path_to_data_dir, args.imgs_dir, args.msks_dir]


def count_dirs_sbdirs(dirs):
    for dr in dirs:
        print(f'{dr}: {len(os.listdir(dr))}')
    for patient in os.listdir(dirs[1]):
        init_num = len(os.listdir(os.path.join(dirs[0], patient)))

        img_num = len(os.listdir(os.path.join(dirs[1], patient)))
        msk_num = len(os.listdir(os.path.join(dirs[2], patient)))
        if not(init_num == img_num+1 and msk_num == 1):
            print(patient)
            print(f'init_num: {init_num}; img_num: {img_num}, msk_num: {msk_num}')







def get_files_to_parse_generator(dir_path):
    dir_gen = os.walk(dir_path)
    for cur_dir, sub_dirs, _ in dir_gen:
        correct_modes = [subdir for subdir in sub_dirs if subdir in IMAGES_MODES]
        if len(correct_modes) > 0:
            yield cur_dir, correct_modes


def convert_masks_to_porper_ext(dir, ext_to_set='.png'):
    if ext_to_set is None:
        return
    for par_dir, _, files in os.walk(dir):
        for f in files:
            if f.endswith('.dcm.lbl'):
                path_to_copy_to = os.path.join(par_dir, f[:-8]+ext_to_set)
                shutil.move(os.path.join(par_dir, f), path_to_copy_to)


def get_nii_sizes(files_gen, source_dir, msk_ext='.png'):
    dict_to_return = {}
    #source_dir_name = os.path.split(source_dir)[1]
    for parent_path, channels in files_gen:

        for channel in channels:
            channel_path = os.path.join(parent_path, channel)

            channel_subdirs_names = [k for k in os.listdir(channel_path) if os.path.isdir(os.path.join(channel_path, k))]

            for ch_sbdir in channel_subdirs_names:
                #assert len(filtered_channel_subdirs_names) == 1
                ch_sbdir_path = os.path.join(channel_path, ch_sbdir)

                files_names = os.listdir(ch_sbdir_path)

                imgs, msks = [], []
                #if len(files_names) > 0 and check_files_format(files_names, ['.dcm', msk_ext]):
                if len(files_names) > 0:
                    files_names = [f_n for f_n in files_names if f_n.endswith('.dcm') or f_n.endswith(msk_ext)]
                    for i in range(len(files_names)):
                        file_name = files_names[i]
                        file_path = os.path.join(ch_sbdir_path, file_name)
                        if file_name.endswith(msk_ext):
                            msks.append(np.asarray(imageio.imread(file_path)))
                            continue

                        try:
                            imgs.append(pydicom.dcmread(os.path.join(ch_sbdir_path, file_name)).pixel_array)
                        except NotImplementedError as e:
                            continue

                imgs = np.asarray(imgs)
                msks = np.asarray(msks)
                pth_to_print = str.replace(ch_sbdir_path, "D:\Универы\ИТМО\Thesis\MRI scans\data\BurtsevLab_data_cleaned\\", '')
                #dict_to_return[pth_to_print] = {'img': imgs.shape, 'msks': msks.shape}
                print(f'{pth_to_print}: imgs shape: {imgs.shape}, masks shape: {msks.shape}')
                del imgs, msks
                gc.collect()
    return dict_to_return



def check_files_format(file_names_to_ckeck, formats=('.dcm',)):
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
