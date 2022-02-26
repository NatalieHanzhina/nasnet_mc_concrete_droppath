import nibabel as nib
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default='data_nii_gz')
    parser.add_argument('--output_dir', default='data_nii')
    args = parser.parse_args()
    perform_walk(args.input_dir, args.output_dir)


def perform_walk(source_dir, target_dir):
    for rt_dir, _, file_names in os.walk(source_dir):
        for file_name in file_names:
            if file_name.endswith('.nii.gz'):
                os.makedirs(rt_dir.replace(source_dir, target_dir), exist_ok=True)
                file_path = os.path.join(rt_dir, file_name)
                nib_fs = nib.load(file_path)
                nib.save(nib_fs, file_path.replace(source_dir, target_dir)[:-7])


if __name__ == '__main__':
    main()
