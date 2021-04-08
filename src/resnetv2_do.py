# -*- coding: utf-8 -*-
"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation (which has some additional
layers and different number of filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import numpy as np
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs

from models import NetType
from models.nasnet_utils_do import ScheduledDropout
from resnetv2_utils_do import DropPath

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block_do(x, scale, block_type, block_idx, cell_num, total_num_cells, total_training_steps,
                              activation='relu', net_type=NetType.mc, do_p=0.3):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](keras./activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        if net_type == NetType.sdp:
            branch_0 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_0)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        if net_type == NetType.sdp:
            branch_1 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_1)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        if net_type == NetType.sdp:
            branch_2 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_2)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        if net_type == NetType.sdp:
            branch_0 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_0)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        if net_type == NetType.sdp:
            branch_1 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_1)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        if net_type == NetType.sdp:
            branch_0 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_0)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        if net_type == NetType.sdp:
            branch_1 = ScheduledDropout(do_p, cell_num=cell_num, total_num_cells=total_num_cells,
                                        total_training_steps=total_training_steps)(
                branch_1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    if net_type == NetType.mc_dp:
        dropped_branches = DropPath(do_p, [True,]*len(branches), name=block_name + '_droppath')(branches, training=True)
        branches = dropped_branches

    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](keras./activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def rand_one_in_array(dropout_len, seed=None):
    rand_arr = tf.zeros(dropout_len)
    rand_arr[tf.random.uniform(1, max_val=dropout_len)] = 1
    return rand_arr


def InceptionResNetV2Same_do(include_top=True,
                             weights='imagenet',
                             input_tensor=None,
                             input_shape=None,
                             total_training_steps=None,
                             pooling=None,
                             net_type=NetType.vanilla,
                             do_p=0.3,
                             classes=1000):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.

    The model and the weights are compatible with TensorFlow, Theano and
    CNTK backends. The data format convention used by the model is
    the one specified in your Keras config file.

    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=None)  # weights=None to prevent input channels equality check

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    total_num_cells = 43
    cell_counter = 0
    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='same')
    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3)
    conv1 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    conv2 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    if net_type == NetType.sdp:
        branch_0 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(
            branch_0)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    if net_type == NetType.sdp:
        branch_1 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(
            branch_1)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    if net_type == NetType.sdp:
        branch_2 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(
            branch_2)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    if net_type == NetType.mc_dp:
        # print('MC_DP___________')
        dropped_branches = DropPath(do_p, [True, True, True, False], name='inception-a_block_droppath')(branches, training=True)
        branches = dropped_branches
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)
    cell_counter += 1

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block_do(x,
                                      scale=0.17,
                                      block_type='block35',
                                      block_idx=block_idx,
                                      cell_num=cell_counter + block_idx,
                                      total_num_cells=total_num_cells,
                                      total_training_steps=total_training_steps,
                                      net_type=net_type,
                                      do_p=do_p)
    conv3 = x
    cell_counter += 10
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    if net_type == NetType.sdp:
        branch_0 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(branch_0)
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    if net_type == NetType.sdp:
        branch_1 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(branch_1)
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    if net_type == NetType.mc_dp:
        dropped_branches = DropPath(do_p, [True, True, False], name='reduction-a_block_droppath')(branches, training=True)
        branches = dropped_branches
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)
    cell_counter += 1

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block_do(x,
                                      scale=0.1,
                                      block_type='block17',
                                      block_idx=block_idx,
                                      cell_num=cell_counter + block_idx,
                                      total_num_cells=total_num_cells,
                                      total_training_steps=total_training_steps,
                                      net_type=net_type,
                                      do_p=do_p)
    conv4 = x
    cell_counter += 20
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    if net_type == NetType.sdp:
        branch_0 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(branch_0)
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    if net_type == NetType.sdp:
        branch_1 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(
            branch_1)
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    if net_type == NetType.sdp:
        branch_2 = ScheduledDropout(do_p, cell_num=cell_counter, total_num_cells=total_num_cells,
                                    total_training_steps=total_training_steps)(
            branch_2)
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    if net_type == NetType.mc_dp:
        dropped_branches = DropPath(do_p, [True, True, True, False], name='reduction-b_block_droppath')(branches, training=True)
        branches = dropped_branches
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)
    cell_counter += 1

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block_do(x,
                                      scale=0.2,
                                      block_type='block8',
                                      block_idx=block_idx,
                                      cell_num=cell_counter + block_idx,
                                      total_num_cells=total_num_cells,
                                      total_training_steps=total_training_steps,
                                      net_type=net_type,
                                      do_p=do_p)
    cell_counter += 9
    x = inception_resnet_block_do(x,
                                  scale=1.,
                                  activation=None,
                                  block_type='block8',
                                  block_idx=10,
                                  cell_num=cell_counter + block_idx,
                                  total_num_cells=total_num_cells,
                                  total_training_steps=total_training_steps,
                                  net_type=net_type,
                                  do_p=do_p)
    cell_counter += 1
    print(f'TOTAL NUMBER OF CELLS ={cell_counter}')

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    conv5 = x
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, [conv1, conv2, conv3, conv4, conv5], name='inception_resnet_v2')
    # Create donor model
    if input_shape[-1] > 3 and weights is not None:
        donor_input_shape = (*input_shape[:-1], 3)
        donor_model = get_donor_model(include_top, input_tensor=None,
                                      input_shape=donor_input_shape,
                                      pooling=pooling,
                                      classes=classes)

    # Load weights
    if weights is not None and input_shape[-1] > 3:
        if weights == 'imagenet':
            if include_top:
                print('Loading pretrained ImageNet weights, include top for inception_resnet_v2 backbone')
                weights_path = get_file('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash='e693bd0210a403b3192acc6073ad2e96')
            else:
                print('Loading pretrained ImageNet weights, exclude top for inception_resnet_v2 backbone')
                weights_path = get_file('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='d19885ff4a710c122648d3b5c3b684e4')
        else:
            ValueError('This is an unexpected value for "weights" parameter')
        if input_shape[-1] > 3:
            print(f'Copying pretrained ImageNet weights to model with {input_shape[-1]} input channels '
                  f'for inception_resnet_v2 backbone')
            donor_model.load_weights(weights_path)

            j = 1 # ignore input layers
            for i, l in enumerate(model.layers[1:]):
                if j >= len(donor_model.layers):
                    break
                d_l = donor_model.layers[j]
                #if l.name != d_l.name: # incorrect names
                if ('dropout' in l.name or 'droppath' in l.name) and ('dropout' not in d_l.name or 'droppath' not in d_l.name):
                    continue
                j += 1
                if \
                        i == 0:
                    new_w = tf.tile(d_l.weights[0], (1, 1, 2, 1))[:, :, :input_shape[-1], :]
                    l.weights[0].assign(new_w)
                else:
                    for (w, d_w) in zip (l.weights, d_l.weights):
                        w.assign(d_w)
            assert j == len(donor_model.layers)

            if weights != 'imagenet':
                print(f'Loading trained "{weights}" weights')
                f = h5py.File(weights, 'r')
                for i, l in enumerate(model.layers):
                    l_ws = l.weights
                    #print(len(f.keys()))
                    #for k in f.keys():
                    #    print(k)
                    #input()
                    d_ws = [f[l.name][l_w.name] for l_w in l_ws]
                    if i == 1:
                        new_w = np.concatenate((d_ws[0].value, l.weights[0].numpy()[..., 3:, :]), axis=-2)
                        l.weights[0].assign(new_w)
                        continue
                    for (w, d_w) in zip(l.weights, d_ws):
                        w.assign(d_w.value)
            del donor_model
        else:
            model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print('No pretrained weights passed')

    return model


def get_donor_model(include_top=True,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000):
    """Instantiates the Inception-ResNet v2 donor architecture."""

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='same')
    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3)
    conv1 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    conv2 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                      scale=0.17,
                                      block_type='block35',
                                      block_idx=block_idx)
    conv3 = x
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                      scale=0.1,
                                      block_type='block17',
                                      block_idx=block_idx)
    conv4 = x
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                      scale=0.2,
                                      block_type='block8',
                                      block_idx=block_idx)
    x = inception_resnet_block(x,
                                  scale=1.,
                                  activation=None,
                                  block_type='block8',
                                  block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    conv5 = x
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(inputs, [conv1, conv2, conv3, conv4, conv5], name='donor_inception_resnet_v2')


if __name__ == '__main__':
    InceptionResNetV2Same_do(include_top=False, input_shape=(256, 256, 3)).summary()
