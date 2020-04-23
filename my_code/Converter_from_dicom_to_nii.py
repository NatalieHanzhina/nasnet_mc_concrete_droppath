import os
from argparse import ArgumentParser

import pydicom
import numpy as np
import nibabel as nib


IMAGES_MODES = ['ML_DWI', 'ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']


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
    for cur_dir, _, _ in dir_gen:
        if os.path.split(cur_dir)[1] in IMAGES_MODES:
            yield cur_dir


def open_and_write_files_to_parse(files_gen, source_dir, save_dir_name):
    #source_dir_name = os.path.split(source_dir)[1]
    for dir_path in files_gen:
        #save_dir_path = str.replace(dir_path, source_dir_name, save_dir_name)
        save_dir_path = str.replace(dir_path, source_dir, save_dir_name)
        for cur_dir, sub_dirs, files_names in os.walk(dir_path):
            if len(sub_dirs) == 0 and check_files_format(files_names):
                dir_to_make = cur_dir.replace(source_dir[:-1], save_dir_name)
                dir_to_make = dir_to_make.replace(dir_to_make.split('/')[-1], '')
                os.makedirs(dir_to_make, exist_ok=True)
                read_files = [pydicom.dcmread(os.path.join(cur_dir, f_n)).pixel_array for f_n in files_names if f_n.endswith('.dcm')]
                nib_imgs = nib.Nifti1Image(np.asarray(read_files), np.eye(4))
                nib.save(nib_imgs, str.replace(cur_dir, source_dir[:-1], save_dir_name)+'.nii.gz')


def check_files_format(file_names_to_ckeck, format='dcm'):
    for f in file_names_to_ckeck:
        if not f.endswith(format if '.' in format else'.' + format):
            return False
    return True


if __name__ == '__main__':
    main()
