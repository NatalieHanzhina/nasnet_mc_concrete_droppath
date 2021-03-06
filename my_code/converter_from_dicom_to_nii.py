import os
import shutil
from argparse import ArgumentParser
from shutil import rmtree

import cv2
import imageio
import nibabel as nib
import numpy as np
import pydicom
from tqdm import tqdm

# MRI_MODES = ['ML_DWI', 'ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
MRI_MODES = ['ML_T1', 'ML_T1+C', 'ML_T2', 'ML_T2_FLAIR']
MRI_AXE = 'Ax'
EXCLUDED_DIR_COUNTER = 0


def main():
    data_dir, save_path, resize_size = get_args()
    convert_masks_to_porper_ext(data_dir)
    open_and_write_files_to_parse(data_dir, save_path, resize_size)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_data_dir', help='Path to data dir to parse')
    parser.add_argument('--save_path', help='Path to data dir to save results', default='converted_nii_gz')
    parser.add_argument('--resize_size', default=256)
    args = parser.parse_args()
    return args.path_to_data_dir, args.save_path, args.resize_size


def convert_masks_to_porper_ext(dir, ext_to_set='.png'):
    if ext_to_set is None:
        return
    for par_dir, _, files in os.walk(dir):
        for f in files:
            if f.endswith('.dcm.lbl'):
                path_to_copy_to = os.path.join(par_dir, f[:-8]+ext_to_set)
                shutil.move(os.path.join(par_dir, f), path_to_copy_to)



def open_and_write_files_to_parse(source_dir, save_dir_name, resize_size):
    global EXCLUDED_DIR_COUNTER
    shutil.rmtree(save_dir_name, ignore_errors=True)

    for patient in tqdm(os.listdir(source_dir)):
        patient_path = os.path.join(source_dir, patient)

        channels_dcm_to_write = dict()
        labels_to_write = dict()
        channel_inconsistency = False
        for channel in os.listdir(patient_path):
            if channel not in MRI_MODES or channel_inconsistency:
                continue
            channel_path = os.path.join(patient_path, channel)

            modes = os.listdir(channel_path)
            if len(modes) == 0:
                print(f'{patient}/{channel} empty channel')
                channel_inconsistency = True
                break
            if len(modes) > 1:
                clean_modes = [m for m in modes if str.lower(MRI_AXE) in str.lower(m)]    # MRI_AXE is in mode name
                if len(clean_modes) > 1:
                    clean_modes = [m for m in modes if str.lower(m).startswith(str.lower(MRI_AXE).strip())]   # mode name starts from MRI_AXE
                if len(clean_modes) != 1:
                    print(f'{patient}/{channel}: {clean_modes}  {len(clean_modes)} CLEAN NODES')
                    channel_inconsistency = True
                    break
                modes = clean_modes
            mode = modes[0]
            mode_path = os.path.join(channel_path, mode)

            dcm_names = []
            mode_dcms = []
            mode_lbls = []
            f_counter = 1

            #print(sorted(os.listdir(mode_path)))
            for f_name in sorted(os.listdir(mode_path)):
                if not(str(f_counter)+'.' in f_name or str(f_counter-1)+'.png' in f_name):      # check for right files naming order inside mode
                    print(f'{patient}/{channel}')
                    print(f'{str(f_counter)}. expected to appear in name, got {f_name} instead')
                    channel_inconsistency = True
                    break
                f_path = os.path.join(mode_path, f_name)
                if f_name.endswith('.dcm'):
                    mode_dcms.append(pydicom.read_file(f_path))
                    dcm_names.append(f_name)
                    f_counter += 1

                elif f_name.endswith('.png'):
                    lbl = imageio.imread(f_path)
                    while len(mode_lbls) < f_counter:
                        mode_lbls.append(np.zeros(lbl.shape))
                    mode_lbls.append(lbl)

            if len(mode_lbls) > 0:                                          # pad nonzero lbls to fit scans shape
                mode_lbls = mode_lbls[2:]
                last_non_zero_lbl_shape = mode_lbls[-1].shape
                while len(mode_lbls) < len(mode_dcms):
                    mode_lbls.append(np.zeros(last_non_zero_lbl_shape))

            for i in range(1, len(mode_dcms)):                              # check z coordinate order in scans
                if not mode_dcms[i][0x20, 0x32][2] > mode_dcms[i-1][0x20, 0x32][2]:
                    print(f'{patient}/{channel}/{mode}: {dcm_names[i-1]} >= {dcm_names[i]}:  '
                          f'{mode_dcms[i-1][0x20, 0x32][2]} >= {mode_dcms[i][0x20, 0x32][2]} ')

            try:
                np.asarray([m.pixel_array for m in mode_dcms])
            except NotImplementedError:
                print(f'{patient}/{channel} NotImplementedError')
                channel_inconsistency=True
                break
            channels_dcm_to_write[channel] = mode_dcms

            if len(mode_lbls) > 0:
                mode_lbls_ndarray = np.asarray(mode_lbls)
                prepr_mode_lbls_ndarray = preprocess_labels(mode_lbls_ndarray, resize_size)
                if prepr_mode_lbls_ndarray.max() > 0:
                    labels_to_write[channel] = prepr_mode_lbls_ndarray

        if channel_inconsistency:
            EXCLUDED_DIR_COUNTER += 1
            continue

        scans_numbers_to_take_set = set(len(slice_list) for slice_list in channels_dcm_to_write.values())
        if len(scans_numbers_to_take_set) != 1:
            channels_dcm_to_write_new, labels_to_write = match_scans(channels_dcm_to_write, labels_to_write)

            if len(channels_dcm_to_write_new.keys()) < len(MRI_MODES) or len(labels_to_write.keys()) == 0:
                print(f'{patient} channel or labels scans_numbers are inconsistent, could not fix')
                EXCLUDED_DIR_COUNTER += 1
                continue
            channels_dcm_to_write = channels_dcm_to_write_new

        if len(labels_to_write.keys()) == 0:
            print(f'{patient}: No lbls detected')
            EXCLUDED_DIR_COUNTER += 1
            continue


        scans_number_to_take = scans_numbers_to_take_set.pop()
        os.makedirs(os.path.join(save_dir_name, 'images', patient), exist_ok=True)
        for channel_name in channels_dcm_to_write.keys():
            ch_path_to_save = os.path.join(save_dir_name, 'images', patient, channel_name)+'.nii'
            scans_to_write = np.asarray(list(c.pixel_array for c in channels_dcm_to_write[channel_name][:scans_number_to_take]))
            data_to_write = preprocess_channel(scans_to_write, resize_size).transpose(1,2,0)
            nib_ch_to_save = nib.Nifti1Image(data_to_write, np.eye(4))
            print(f'Saving channel shape {nib_ch_to_save.get_fdata().shape}')
            if len(nib_ch_to_save.get_fdata().shape) != 3:
                input()
            nib.save(nib_ch_to_save, ch_path_to_save)

        if len(labels_to_write.keys()) > 1:  # print info about labels
            print('More than lbls detected')
            for k in labels_to_write.keys():
                print(f'{k}: {np.unique(labels_to_write[k], return_counts=True)}')

        os.makedirs(os.path.join(save_dir_name, 'masks', patient), exist_ok=True)
        lbl_name_to_take = list(labels_to_write.keys())[-1]
        lbl_path_to_save = os.path.join(save_dir_name, 'masks', patient, lbl_name_to_take)+'_seg.nii'
        nib_lbl_to_save = nib.Nifti1Image(np.asarray(labels_to_write[lbl_name_to_take]).transpose(1,2,0), np.eye(4))
        print(f'Saving msk shape {nib_lbl_to_save.get_fdata().shape}')
        if len(nib_lbl_to_save.get_fdata().shape) != 3:
            input()
        nib.save(nib_lbl_to_save, lbl_path_to_save)

        a = 1

    print(f'{EXCLUDED_DIR_COUNTER} EXCLUDED DIRS')


def preprocess_channel(channel_data, resize_size):
    if resize_size is None:
        return channel_data
    resized_img = []
    for i in range(channel_data.shape[0]):
        img = channel_data[i]
        resized_img.append(cv2.resize(img, tuple(min(img.shape[j], resize_size) for j in range(len(img.shape)))))

    ndarray_img = np.asarray(resized_img)
    ndarray_img = ndarray_img.transpose(0, 2, 1)
    return ndarray_img

def preprocess_labels(mask, resize_size):
    if resize_size is None:
        return np.where(mask == 2, 1, 0)

    resized_mask = []
    transf_msk = np.where(mask == 2, 1, 0)
    for i in range(transf_msk.shape[0]):
        cur_msk = transf_msk[i]
        if cur_msk.max() == 0:
            resized_mask.append(np.zeros(tuple(min(cur_msk.shape[j], resize_size) for j in range(len(cur_msk.shape)))))
        else:
            resized_mask.append(cv2.resize(cur_msk.astype('int16'), tuple(min(cur_msk.shape[j], resize_size) for j in range(len(cur_msk.shape))), cv2.INTER_LINEAR))

    resized_mask = np.asarray(resized_mask)
    resized_mask = resized_mask[..., ::-1]
    ndarray_mask = resized_mask.transpose(0, 2, 1)
    return ndarray_mask


def match_scans(channels_to_match, labels_to_match):
    labels_z_coords = {k: [] for k in labels_to_match.keys()}
    for l_k in labels_to_match.keys():
        for k in channels_to_match.keys():
            if len(channels_to_match[k]) == len(labels_to_match[l_k]):
                labels_z_coords[l_k] = [slice[0x20, 0x32][2] for slice in channels_to_match[k]]
                break
    min_scans_numbers_to_take = min(len(slice_list)for slice_list in channels_to_match.values())
    z_coords_to_take = [float(dcm[0x20, 0x32][2]) for dcm in min(channels_to_match.items(), key=lambda x: len(x[1]))[1]]

    channel_slices_to_take = dict()
    channel_names_to_fill = list(channels_to_match.keys())
    i = -1
    while i < min_scans_numbers_to_take + 10:
        i += 1
        slices_to_take = {}
        for k in channel_names_to_fill:
            if len(channels_to_match[k]) > i and float(channels_to_match[k][i][0x20, 0x32][2]) in z_coords_to_take:
                slices_to_take[k] = channels_to_match[k][i]

        for k in slices_to_take:
            if k not in channel_slices_to_take.keys():
                channel_slices_to_take[k] = []
            channel_slices_to_take[k].append(slices_to_take[k])
        channel_names_to_fill = list(k for k in channel_names_to_fill if k not in channel_slices_to_take.keys() or
                                     len(channel_slices_to_take[k]) < min_scans_numbers_to_take)

    final_z_coords_to_take = [float(dcm[0x20, 0x32][2]) for dcm in min(channel_slices_to_take.items(), key=lambda x: len(x[1]))[1]]

    final_channel_slices_to_take = {k: [] for k in channel_slices_to_take}
    for channel in channel_slices_to_take.keys():
        final_channel_lst = []
        for i in range(len(channel_slices_to_take[channel])):
            if float(channel_slices_to_take[channel][i][0x20, 0x32][2]) in final_z_coords_to_take:
                final_channel_lst.append(channel_slices_to_take[channel][i])
        final_channel_slices_to_take[channel] = final_channel_lst

    final_labels_to_take = dict()
    for l_k in labels_to_match.keys():
        for i in range(len(labels_z_coords[l_k])):
            l_coord = labels_z_coords[l_k][i]
            if float(l_coord) in z_coords_to_take:
                if l_k not in final_labels_to_take.keys():
                    final_labels_to_take[l_k] = []
                final_labels_to_take[l_k].append(labels_to_match[l_k][i])

    return final_channel_slices_to_take, final_labels_to_take


if __name__ == '__main__':
    main()

