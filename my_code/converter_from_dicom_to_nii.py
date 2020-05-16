import numbers
import os
import shutil
import time
from argparse import ArgumentParser

import imageio
import nibabel as nib
import numpy as np
import pydicom
import pydicom.uid
from tqdm import tqdm

#IMAGES_MODES = ['ML_DWI', 'ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
IMAGES_MODES = ['ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
DIR_FILTER_PREFIX = 'Ax'
DIR_FILTER_NAME = 'AX'
MSK_SHAPE = (512, 512)

EXCLUDED_DIRS = 0


def main():
    data_dir, save_path = get_args()
    #convert_masks_to_porper_ext(data_dir)
    files_to_parse_gen = get_files_to_parse_generator(data_dir)
    #check_sm_shit(files_to_parse_gen, data_dir, save_path)
    open_and_write_files_to_parse(files_to_parse_gen, data_dir, save_path)
    print(f'Excluded {EXCLUDED_DIRS} directories')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_data_dir', help='Path to data dir to parse')
    parser.add_argument('--save_path', help='Path to data dir to save results', default='converted_nii_gz')
    args = parser.parse_args()
    return args.path_to_data_dir, args.save_path


def get_files_to_parse_generator(dir_path):
    patients = os.listdir(dir_path)
    for patient in tqdm(patients):
        patient_path = os.path.join(dir_path, patient)
        sub_dirs = os.listdir(patient_path)
        correct_modes = [subdir for subdir in sub_dirs if subdir in IMAGES_MODES]
        if len(correct_modes) > 0:
            yield patient_path, correct_modes


def convert_masks_to_porper_ext(dir, ext_to_set='.png'):
    if ext_to_set is None:
        return
    for par_dir, _, files in os.walk(dir):
        for f in files:
            if f.endswith('.dcm.lbl'):
                path_to_copy_to = os.path.join(par_dir, f[:-8]+ext_to_set)
                shutil.move(os.path.join(par_dir, f), path_to_copy_to)


def check_sm_shit(files_gen, source_dir, save_dir_path, msk_ext='.png'):
    #source_dir_name = os.path.split(source_dir)[1]
    for parent_path, channels in files_gen:
        if source_dir[-1] == '/':
            save_msk_path = str.replace(parent_path, source_dir[:-1], os.path.join(save_dir_path, 'masks'))
        else:
            save_msk_path = str.replace(parent_path, source_dir, os.path.join(save_dir_path, 'masks'))
        #save_msk_path = str.replace(parent_path, source_dir[:-1], os.path.join(save_dir_path, 'masks'))
        save_msk_path = os.path.join(save_msk_path, 'seg')

        for channel in channels:
            channel_path = os.path.join(parent_path, channel)
            if source_dir[-1] == '/':
                save_img_channel_path = str.replace(channel_path, source_dir[:-1], os.path.join(save_dir_path, 'images'))
            else:
                save_img_channel_path = str.replace(channel_path, source_dir, os.path.join(save_dir_path, 'images'))


            channel_subdirs_names = [k for k in os.listdir(channel_path) if os.path.isdir(os.path.join(channel_path, k))]
            if len(channel_subdirs_names) > 1:
                filtered_channel_subdirs_names = [d for d in channel_subdirs_names if DIR_FILTER_PREFIX in d or
                                                  d == DIR_FILTER_NAME]
            else:
                filtered_channel_subdirs_names = channel_subdirs_names

            if len(filtered_channel_subdirs_names) != 1:
                pt = str.replace(channel_path, "D:\Универы\ИТМО\Thesis\MRI scans\data\BurtsevLab_data_cleaned\\", "")
                print(f'{pt} contains {len(filtered_channel_subdirs_names)} suitable dirs!')
                if len(filtered_channel_subdirs_names) == 0:
                    continue

            #assert len(filtered_channel_subdirs_names) == 1
            ch_sbdir = filtered_channel_subdirs_names[0]
            ch_sbdir_path = os.path.join(channel_path, ch_sbdir)


            files_names = os.listdir(ch_sbdir_path)

            #if len(files_names) > 0 and check_files_format(files_names, ['.dcm', msk_ext]):
            if len(files_names) > 0:
                files_names = [f_n for f_n in files_names if f_n.endswith('.dcm') or f_n.endswith(msk_ext)]
                non_zero_msk_counter = 0
                mask_fs = []
                img_files = []
                for i in range(len(files_names)):
                    msk_shape = None
                    file_name = files_names[i]
                    file_path = os.path.join(ch_sbdir_path, file_name)
                    if file_name.endswith(msk_ext):
                        mask_fs.append(imageio.imread(file_path))
                        non_zero_msk_counter += 1
                        msk_shape = mask_fs[-1].shape
                        continue

                    try:
                        img_files.append(pydicom.dcmread(os.path.join(ch_sbdir_path, file_name)).pixel_array)
                    except NotImplementedError as e:
                        pt = str.replace(os.path.join(ch_sbdir_path, file_name), "D:\Универы\ИТМО\Thesis\MRI scans\data\BurtsevLab_data_cleaned\\", "")
                        print(f'{pt} gives "{e}"!')
                        global EXCLUDED_DIRS
                        EXCLUDED_DIRS += 1
                        continue
                    msk_shape = img_files[-1].shape if msk_shape is None else msk_shape


def open_and_write_files_to_parse(files_gen, source_dir, save_dir_path, msk_ext='.png'):
    global EXCLUDED_DIRS
    for patient_dir, channels in files_gen:
        masks = {}
        channels_imgs = {}
        patient_img_shape = None
        indices_to_exclude = set()

        channel_fail = False
        for channel in channels:
            if channel_fail:
                continue
            channel_path = os.path.join(patient_dir, channel)

            channel_subdirs_names = [k for k in os.listdir(channel_path) if os.path.isdir(os.path.join(channel_path, k))]
            if len(channel_subdirs_names) > 1:
                filtered_channel_subdirs_names = [d for d in channel_subdirs_names if DIR_FILTER_PREFIX in d or
                                                  d == DIR_FILTER_NAME]
            else:
                filtered_channel_subdirs_names = channel_subdirs_names

            if len(filtered_channel_subdirs_names) != 1:
                channel_fail = True
                EXCLUDED_DIRS += 1
                continue

            assert len(filtered_channel_subdirs_names) == 1
            ch_sbdir = filtered_channel_subdirs_names[0]
            ch_sbdir_path = os.path.join(channel_path, ch_sbdir)

            files_names = os.listdir(ch_sbdir_path)
            first_file_name = [f_n for f_n in files_names if f_n.endswith('.dcm') or f_n.endswith(msk_ext)][0]
            if patient_img_shape is None:
                patient_img_shape = pydicom.dcmread(os.path.join(ch_sbdir_path, first_file_name)).pixel_array.shape
            else:
                current_img_shape = pydicom.dcmread(os.path.join(ch_sbdir_path, first_file_name)).pixel_array.shape
                if current_img_shape != patient_img_shape:
                    channel_fail = True
                    EXCLUDED_DIRS += 1
                    continue


            #if len(files_names) > 0 and check_files_format(files_names, ['.dcm', msk_ext]):
            if len(files_names) > 0:
                files_names = [f_n for f_n in files_names if f_n.endswith('.dcm') or f_n.endswith(msk_ext)]
                non_zero_msk_counter = 0
                mask_fs = []
                img_files = []
                for i in range(len(files_names)):
                    file_name = files_names[i]
                    file_path = os.path.join(ch_sbdir_path, file_name)
                    if file_name.endswith(msk_ext):
                        mask = np.asarray(imageio.imread(file_path))
                        pr_mask = preprocess_mask(mask)
                        mask_fs.append(pr_mask)
                        non_zero_msk_counter += 1
                        continue

                    try:
                        img_files.append(pydicom.dcmread(os.path.join(ch_sbdir_path, file_name)).pixel_array)
                    except NotImplementedError:
                        indices_to_exclude.add(i)
                    if i >= len(files_names)-1 or not files_names[i+1].endswith(msk_ext):
                        mask_fs.append(0)

                if non_zero_msk_counter > 0:
                    masks[channel] = mask_fs

                channels_imgs[channel] = img_files

        if channel_fail:
            continue
        min_img_number = 10000
        for channel in channels:
            channel_path = os.path.join(patient_dir, channel)
            if source_dir[-1] == '/':
                save_img_channel_path = str.replace(channel_path, source_dir[:-1],
                                                    os.path.join(save_dir_path, 'images'))
            else:
                save_img_channel_path = str.replace(channel_path, source_dir, os.path.join(save_dir_path, 'images'))
            shutil.rmtree(save_img_channel_path+'.nii', ignore_errors=True)
            os.makedirs(os.path.split(save_img_channel_path)[0], exist_ok=True)

            img_files = channels_imgs[channel]
            channel_indices_to_exclude = [i for i in indices_to_exclude if i <= len(img_files)]
            if len(channel_indices_to_exclude) > 0:
                img_files = np.delete(img_files, tuple(channel_indices_to_exclude), axis=0)
            min_img_number = min(min_img_number, np.asarray(img_files).shape[0])
            img_files_t = np.asarray(img_files).transpose((1,2,0))
            nib_imgs = nib.Nifti1Image(img_files_t, np.eye(4))
            nib.save(nib_imgs, save_img_channel_path + '.nii')

        if source_dir[-1] == '/':
            save_msk_path = str.replace(patient_dir, source_dir[:-1], os.path.join(save_dir_path, 'masks'))
        else:
            save_msk_path = str.replace(patient_dir, source_dir, os.path.join(save_dir_path, 'masks'))
        save_msk_path = os.path.join(save_msk_path, 'seg')
        shutil.rmtree(os.path.split(save_msk_path)[0], ignore_errors=True)
        time.sleep(0.3)
        os.makedirs(os.path.split(save_msk_path)[0])

        masks_to_take = []
        masks_stack = list(masks.values())
        for i in range(min_img_number):
            cur_m_stack = [msks[i] for msks in masks_stack]
            chech_for_zeros = np.asarray([isinstance(msk, numbers.Number) or msk.shape != MSK_SHAPE for msk in cur_m_stack])
            if np.all(chech_for_zeros):
                masks_to_take.append(np.zeros(MSK_SHAPE))
            else:
                masks_to_take.append(masks_stack[np.argmax(chech_for_zeros == False)][i])
        msk_indices_to_exclude = [i for i in indices_to_exclude if i <= len(masks_to_take)]
        if len(msk_indices_to_exclude) > 0:
            masks_to_take = np.delete(masks_to_take, tuple(msk_indices_to_exclude), axis=0)
        masks_to_take_t = np.asarray(masks_to_take).transpose((1,2,0))
        nib_imgs = nib.Nifti1Image(np.asarray(masks_to_take_t), np.eye(4))
        nib.save(nib_imgs, save_msk_path + '.nii')


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


def preprocess_mask(mask):
    return np.where(mask == 2, 1, 0)


if __name__ == '__main__':
    main()
