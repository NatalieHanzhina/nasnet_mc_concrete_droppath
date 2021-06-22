"""DenseNet models for Keras.

Reference paper:
  - [Densely Connected Convolutional Networks]
    (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import h5py
import numpy as np
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs

from . import NetType

TF_WEIGHTS_PATH = 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'


def DenseNet169_do(net_type, include_top=True, do_p=0.3, weights='imagenet',
                   input_tensor=None, input_shape=None,
                   pooling=None, classes=1000):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.densenet.preprocess_input` for an example.

    Arguments:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
    A `keras.Model` instance.

    Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """
    blocks = [6, 12, 32, 32]

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

    if K.image_data_format() != 'channels_last':
        warnings.warn('The Densenet169 model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None



    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=None)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    if net_type == NetType.mc:
        x = Dropout(do_p)(x, training=True)
    elif net_type == NetType.mc_df:
        x = Dropout(do_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax',
                     name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='densenet169')
    # Create donor model
    if input_shape[-1] > 3 and weights is not None:
        donor_input_shape = (*input_shape[:-1], 3)
        donor_model = get_donor_model(include_top, input_tensor=None,
                                      input_shape=donor_input_shape,
                                      pooling=pooling,
                                      classes=classes)

    # Load weights.
    if weights is not None and input_shape[-1] > 3:
        if weights == 'imagenet':
            if include_top:
                print('Loading pretrained ImageNet weights, include top for densnet169 backbone')
                #if blocks == [6, 12, 32, 32]:
                weights_path = get_file('densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash='d699b8f76981ab1b30698df4c175e90b')
            else:
                print('Loading pretrained ImageNet weights, exclude top for densnet169 backbone')
                #if blocks == [6, 12, 32, 32]:
                weights_path = get_file('densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
        else:
            ValueError('This is an unexpected value for "weights" parameter')
        if input_shape[-1] > 3:
            print(f'Copying pretrained ImageNet weights to model with {input_shape[-1]} input channels for xception backbone')
            donor_model.load_weights(weights_path)

            donor_model_weight_layers = [d_l for d_l in donor_model.layers if len(d_l.weights) > 0]
            j = 0
            already_copied_layers = []
            for i, l in enumerate([l for l in model.layers if len(l.weights) > 0]):
                if j >= len(donor_model_weight_layers):
                    break
                while j in already_copied_layers:
                    j += 1
                d_l = donor_model_weight_layers[j]
                # if l.name != d_l.name: # incorrect names
                if 'dropout' in l.name and 'dropout' not in d_l.name or \
                        'droppath' in l.name and 'droppath' not in d_l.name:
                    continue
                if i == 0:
                    new_w = tf.tile(d_l.weights[0], (1, 1, 2, 1))[:, :, :input_shape[-1], :]
                    l.weights[0].assign(new_w)
                    j += 1
                elif l.name == d_l.name:
                    for (w, d_w) in zip(l.weights, d_l.weights):
                        w.assign(d_w)
                    j += 1
                else:
                    for k in range(j + 1, len(donor_model_weight_layers)):
                        d_l_next = donor_model_weight_layers[k]
                        if l.name == d_l_next.name:
                            for (w, d_n_w) in zip(l.weights, d_l_next.weights):
                                w.assign(d_n_w)
                            already_copied_layers.append(k)
                            break
                        if k == len(donor_model_weight_layers) - 1:
                            raise ValueError
            assert j == len(donor_model_weight_layers)

            if weights != 'imagenet':
                print(f'Loading trained "{weights}" weights')
                f = h5py.File(weights, 'r')
                for i, l in enumerate(model.layers):
                    l_ws = l.weights
                    # print(len(f.keys()))
                    # for k in f.keys():
                    #    print(k)
                    # input()
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

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def get_donor_model(include_top=True, input_tensor=None,
                    input_shape=None, pooling=None, classes=1000):
    blocks = [6, 12, 32, 32]
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=None)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax',
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(inputs, x, name='densenet169')


def dense_block(x, blocks, name):
    """A dense block.

    Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def dense_block_do(x, blocks, name, net_type, do_p):
    """A dense block.

    Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block_do(x, 32, name=name + '_block' + str(i + 1), net_type=net_type, do_p=do_p)
    return x


def transition_block(x, reduction, name):
    """A transition block.

    Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

    Returns:
    output tensor for the block.
    """
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[-1] * reduction),
                      1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def transition_block_do(x, reduction, name, net_type, do_p=0.3):
    """A transition block.

    Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

    Returns:
    output tensor for the block.
    """
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[-1] * reduction),
                      1,
                      use_bias=False,
                      name=name + '_conv')(x)
    if net_type == NetType.mc:
        x = Dropout(do_p)(x, training=True)
    elif net_type == NetType.mc_df:
        x = Dropout(do_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(name=name + '_concat')([x, x1])
    return x


def conv_block_do(x, growth_rate, name, net_type, do_p):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x1)
    if net_type == NetType.mc:
        x1 = Dropout(do_p)(x1, training=True)
    elif net_type == NetType.mc_df:
        x1 = Dropout(do_p, noise_shape=(x1.shape[0], 1, 1, x1.shape[-1]))(x1, training=True)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    if net_type == NetType.mc:
        x1 = Dropout(do_p)(x1, training=True)
    elif net_type == NetType.mc_df:
        x1 = Dropout(do_p, noise_shape=(x1.shape[0], 1, 1, x1.shape[-1]))(x1, training=True)
    x = layers.Concatenate(name=name + '_concat')([x, x1])
    return x
