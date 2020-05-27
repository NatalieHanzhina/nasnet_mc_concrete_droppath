
import os
import time
from abc import abstractmethod

import cv2
import nibabel as nib
import numpy as np
from params import args
from skimage.color import rgb2gray
from skimage.morphology import square, dilation
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import Iterator, load_img, img_to_array


class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 images_dir,
                 masks_dir,
                 image_ids,
                 images_paths,
                 channels,
                 crop_shape,
                 resize_shape,
                 preprocessing_function,
                 random_transformer=None,
                 batch_size=8,
                 shuffle=True,
                 image_name_template=None,
                 mask_template=None,
                 label_template=None,
                 padding=32,
                 seed=None,
                 grayscale_mask=False,
                 ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = image_ids
        self.image_paths = images_paths
        self.channels = channels
        self.image_name_template = image_name_template
        self.mask_template = mask_template
        self.label_template = label_template
        self.random_transformer = random_transformer
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.preprocessing_function = preprocessing_function
        self.padding = padding
        self.grayscale_mask = grayscale_mask
        self.max_msk_value = self.find_max_mask_value()
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask, image):
        raise NotImplementedError

    def preprocess_mask(self, mask):
        if len(mask.shape) == 3:
            if mask.shape[2] != 3:
                raise NotImplementedError('Such mask dimensions are not supported')
            mask = rgb2gray(mask)
        if len(mask.shape) > 3:
            raise NotImplementedError('Such mask dimensions are not supported')

        # kernel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = square(16)
        bg = dilation(mask, kernel).astype('uint8')
        boarder = np.subtract(bg, mask)
        mask = np.stack((mask, boarder), axis=-1)
        return mask

    def find_max_mask_value(self):
        mask_max = -1e8
        img_paths = list(set(self.image_paths))
        for i in range(len(img_paths)):
            mask_path = img_paths[i].replace(self.images_dir, self.masks_dir)
            if mask_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                return None
            elif mask_path.endswith('.nii.gz') or mask_path.endswith('.nii') or os.path.isdir(mask_path):
                nii_gz_mask_path = os.path.join(mask_path, os.listdir(mask_path)[0])
                if nii_gz_mask_path.endswith('nii.gz') or nii_gz_mask_path.endswith('.nii'):
                    nib_fs = nib.load(nii_gz_mask_path)
                else:
                    nib_fs = nib.load(os.path.join(nii_gz_mask_path, os.listdir(nii_gz_mask_path)[0]))
                masks = nib_fs.get_fdata()
                mask_max = np.max([mask_max, masks.max()])
        return mask_max

    def create_opencv_mask(self, labels):
        if self.grayscale_mask:
            labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)

        tmp = labels.copy()
        tmp = tmp.astype('uint8')

        threshold_level = 127  # Set as you need...
        _, binarized = cv2.threshold(tmp, threshold_level, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        msk1 = np.zeros_like(tmp, dtype='uint8')
        msk1 = cv2.drawContours(msk1, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
        msk2 = np.zeros_like(tmp, dtype='uint8')
        msk = np.stack((labels, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)
        return msk

    def augment_and_crop_mask_image(self, mask, image, label, img_id, crop_shape):
        return mask, image, label

    def transform_batch_y(self, batch_y):
        if self.max_msk_value is None:
            return batch_y
        if self.max_msk_value <= 4:
            return np.where(batch_y < 1, batch_y, 1)
        batch_y[:, -1, -1, -1] = self.max_msk_value
        norm_batch_y = cv2.normalize(batch_y, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        norm_batch_y[:, -1, -1, -1] = norm_batch_y[:, -2, -2, -2]
        batch_y[:, -1, -1, -1] = batch_y[:, -2, -2, -2]
        return norm_batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            id_in_archive = self.image_ids[image_index]

            img_path = self.image_paths[image_index]
            if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                image = np.array(img_to_array(load_img(img_path)), "uint8")
            elif img_path.rfind('.nii.gz') or img_path.rfind('.nii') or os.path.isdir(img_path):
                image = self.read_nii_gz_img_archive(img_path, id_in_archive)
            else:
                raise ValueError("Unsupported type of image input data")

            mask_path = self.image_paths[image_index].replace(self.images_dir, self.masks_dir)
            if mask_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                #mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            elif mask_path.endswith('.nii.gz') or mask_path.endswith('.nii') or os.path.isdir(mask_path):
                mask = self.read_nii_gz_msk_archive(mask_path, id_in_archive)
                mask = mask*255
            else:
                raise ValueError("Unsupported type of mask input data")

            mask = self.create_opencv_mask(mask)
            if args.use_full_masks:
                pass
                #mask[...,0] = (label > 0) * 255
            if self.crop_shape is not None and self.crop_shape != (None, None):
                label = None
                crop_mask, crop_image, crop_label = self.augment_and_crop_mask_image(mask, image, label, id, self.crop_shape)
                data = self.random_transformer(image=np.array(crop_image, "uint8"), mask=np.array(crop_mask, "uint8"))
                crop_image, crop_mask = data['image'], data['mask']
                if len(np.shape(crop_mask)) == 2:
                    crop_mask = np.expand_dims(crop_mask, -1)
                crop_mask = self.transform_mask(crop_mask, crop_image)
                batch_x.append(crop_image)
                batch_y.append(crop_mask)
            elif self.resize_shape is not None and self.resize_shape != (None, None):
                resized_image = cv2.resize(image, tuple(
                    min(image.shape[j], self.resize_shape[j]) for j in range(len(self.resize_shape))))
                resized_mask = cv2.resize(mask, tuple(
                    min(mask.shape[j], self.resize_shape[j]) for j in range(len(self.resize_shape))))
                x0, x1, y0, y1 = 0, 0, 0, 0
                if (resized_image.shape[1] % 32) != 0:
                    x0 = int((32 - resized_image.shape[1] % 32) / 2)
                    x1 = (32 - resized_image.shape[1] % 32) - x0
                if (resized_image.shape[0] % 32) != 0:
                    y0 = int((32 - resized_image.shape[0] % 32) / 2)
                    y1 = (32 - resized_image.shape[0] % 32) - y0
                resized_image = np.pad(resized_image, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                resized_mask = np.pad(resized_mask, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                batch_x.append(resized_image)
                resized_mask = self.transform_mask(resized_mask, resized_image)
                batch_y.append(resized_mask)
            else:
                x0, x1, y0, y1 = 0, 0, 0, 0
                if (image.shape[1] % 32) != 0:
                    x0 = int((32 - image.shape[1] % 32) / 2)
                    x1 = (32 - image.shape[1] % 32) - x0
                if (image.shape[0] % 32) != 0:
                    y0 = int((32 - image.shape[0] % 32) / 2)
                    y1 = (32 - image.shape[0] % 32) - y0
                image = np.pad(image, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                mask = np.pad(mask, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                batch_x.append(image)
                mask = self.transform_mask(mask, image)
                batch_y.append(mask)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        #if self.preprocessing_function:
        #    batch_x = imagenet_utils.preprocess_input(batch_x, mode=self.preprocessing_function)
        #return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)
        t_x, t_y = self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)
        #print(f'transformed img min: {np.min(t_x)}, max: {np.max(t_x)}, init img min: {np.min(batch_x)}, max: {np.max(batch_x)}')
        #print(f'transformed msk min: {np.min(t_y)}, max: {np.max(t_y)}, init msk min: {np.min(batch_y)}, max: {np.max(batch_y)}')

        if self.preprocessing_function and t_x.shape[-1] == 3:
            preprocessed_t_x = imagenet_utils.preprocess_input(t_x, mode=self.preprocessing_function)
        else:
            preprocessed_t_x = t_x
        #print(f'preprocessed transformed img min: {np.min(preprocessed_t_x)}, max: {np.max(preprocessed_t_x)}')
        #input()
        return preprocessed_t_x, t_y

    def read_nii_gz_img_archive(self, img_path, id_in_archive):
        img = []
        for i, channel in enumerate(os.listdir(img_path)):
            if i >= self.channels:
                break
            channel_path = os.path.join(img_path, channel)
            if channel.endswith('nii.gz') or channel.endswith('.nii'):
                nib_fs = nib.load(channel_path)
            else:
                nib_fs = nib.load(os.path.join(channel_path, os.listdir(channel_path)[0]))
            img.append(nib_fs.get_fdata()[..., id_in_archive])
        if len(np.asarray(img).shape) < 3:
            print([k.shape for k in img])
            print(channel_path)
            print(os.listdir(img_path))
            print(np.asarray(img).shape)
            input()
        image = np.asarray(img)
        image = image.transpose((1, 2, 0))
        return image

    @staticmethod
    def read_nii_gz_msk_archive(msk_path, id_in_archive):
        nii_gz_mask_path = os.path.join(msk_path, os.listdir(msk_path)[0])
        if nii_gz_mask_path.endswith('nii.gz') or nii_gz_mask_path.endswith('.nii'):
            nib_fs = nib.load(nii_gz_mask_path)
        else:
            nib_fs = nib.load(os.path.join(nii_gz_mask_path, os.listdir(nii_gz_mask_path)[0]))
        return nib_fs.get_fdata()[..., id_in_archive]

    def transform_batch_x(self, batch_x):
        batch_x_max = batch_x.max(axis=(1,2), keepdims=True)
        norm_batch_x = batch_x*255/batch_x_max

        for l1 in range(norm_batch_x.shape[0]):
            for l2 in range(norm_batch_x.shape[-1]):
                assert norm_batch_x[l1, ..., l2].max() == 255

        if batch_x.shape[-1] > 3:
            mean = [103.939, 116.779, 123.68]
            r_mean = mean[::-1]
            t_n_batch_x = norm_batch_x[..., ::-1] - np.asarray((r_mean+r_mean)[:batch_x.shape[-1]])
            return t_n_batch_x
        return norm_batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


