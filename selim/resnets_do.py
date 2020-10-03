# -*- coding: utf-8 -*-

"""
keras_resnet.models._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional residual models.
"""

import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from models import NetType
from tensorflow import keras
from tensorflow.keras.utils import get_file


resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


def download_resnet_imagenet(v):
    v = int(v.replace('resnet', ''))

    filename = resnet_filename.format(v)
    resource = resnet_resource.format(v)
    if v == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif v == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif v == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def ResNet_do(blocks, block, input_tensor=None, input_shape=None, include_top=True, net_type=NetType.vanilla, do_p=0.3,
              classes=1000, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.

    :param input_shape: optional shape tuple.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than ??.

    :param include_top: if true, includes classification layers

    :param net_type: indicates types of dropout to be used in network MC or MC DP

    :param do_p: dropout rate

    :param classes: number of classes to classify (include_top must be true)


    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet_do(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=keras.backend.image_data_format(),
                                      require_flatten=False,
                                      weights=None)  # weights=None to prevent input channels equality check

    if input_tensor is None:
        inputs = keras.layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            inputs = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor


    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn_conv1")(x)
    if net_type == NetType.mc:
        x = keras.layers.Dropout(do_p)(x, training=True)
    elif net_type == NetType.mc_df:
        x = keras.layers.Dropout(do_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]))(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def donor_ResNet(inputs, blocks, block, include_top=True, classes=1000, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)


    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet_mc(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]))(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def ResNet18(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet18 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet18(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]

    return ResNet_do(inputs, blocks, block=keras_resnet.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet34(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet34 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet34(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return ResNet_do(inputs, blocks, block=keras_resnet.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet_do(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d_dp, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet101 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet101(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 23, 3]
    numerical_names = [False, True, True, False]

    return ResNet_do(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d_dp, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet152(weights, input_tensor=None, input_shape=None, blocks=None, include_top=True, classes=1000,
              *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet152 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet152(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 8, 36, 3]
    numerical_names = [False, True, True, False]

    model = ResNet_do(blocks, input_tensor=input_tensor, input_shape=input_shape, numerical_names=numerical_names,
                     block=bottleneck_2d_dp, include_top=include_top, net_type=NetType.vanilla,
                     classes=classes, *args, **kwargs)

    donor_inputs = keras.layers.Input([*model.input.shape[1:-1], 3])
    donor_model = donor_ResNet(donor_inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d,
                               *args, **kwargs)

    if weights == 'imagenet':
        weights = download_resnet_imagenet("resnet152")

    if weights is not None and input_shape[-1] > 3:
        if input_shape[-1] > 3:
            donor_model.load_weights(weights)
            transfer_weights_from_donor_model(model, donor_model, input_shape[1:])
        else:
            model.load_weights(weights)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No pretrained weights passed")
    return model



def ResNet152_do(weights, net_type, input_tensor=None, input_shape=None, blocks=None, include_top=True, do_p=0.3,
                 classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet152 specifications with MC Dropout or MC dropfilter.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet152(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 8, 36, 3]
    numerical_names = [False, True, True, False]

    model = ResNet_do(blocks, input_tensor=input_tensor, input_shape=input_shape, numerical_names=numerical_names,
                      block=bottleneck_2d_dp, include_top=include_top, net_type=net_type, do_p=do_p, classes=classes,
                      *args, **kwargs)

    donor_inputs = keras.layers.Input([*model.input.shape[1:-1], 3])
    donor_model = donor_ResNet(donor_inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d,
                               *args, **kwargs)

    if weights == 'imagenet':
        weights = download_resnet_imagenet("resnet152")

    if weights is not None and input_shape[-1] > 3:
        if input_shape[-1] > 3:
            donor_model.load_weights(weights)
            transfer_weights_from_donor_model(model, donor_model, input_shape[1:])
        else:
            model.load_weights(weights)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No pretrained weights passed")
    return model


def ResNet152_mc(inputs, weights, blocks=None, include_top=True, do_p=0.3, classes=1000, *args, **kwargs):
    return ResNet152_do(inputs, weights, net_type=NetType.mc, blocks=blocks, include_top=include_top, do_p=do_p,
                        classes=classes, *args, **kwargs)


# def ResNet152_mc_old(inputs, weights, blocks=None, include_top=True, dp_p=0.3, classes=1000, *args, **kwargs):
#     """
#     Constructs a `keras.models.Model` according to the ResNet152 specifications with MC dropout.
#
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#
#     :param blocks: the network’s residual architecture
#
#     :param include_top: if true, includes classification layers
#
#     :param classes: number of classes to classify (include_top must be true)
#
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#
#     Usage:
#
#         >>> import keras_resnet.models
#
#         >>> shape, classes = (224, 224, 3), 1000
#
#         >>> x = keras.layers.Input(shape)
#
#         >>> model = keras_resnet.models.ResNet152(x, classes=classes)
#
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     if blocks is None:
#         blocks = [3, 8, 36, 3]
#     numerical_names = [False, True, True, False]
#
#     model = ResNet_do(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d_dp, include_top=include_top,
#                       net_type=NetType.mc, do_p=dp_p, classes=classes, *args, **kwargs)
#
#     donor_inputs = keras.Input([*inputs.shape[1:-1], 3])
#     donor_model = donor_ResNet(donor_inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d,
#                                *args, **kwargs)
#     donor_model.load_weights(weights)
#
#     transfer_weights_from_donor_model(model, donor_model, inputs.shape[1:])
#     return model


def ResNet152_mc_dp(inputs, weights, blocks=None, include_top=True, do_p=0.3, classes=1000, *args, **kwargs):
    return ResNet152_do(inputs, weights, net_type=NetType.mc_dp, blocks=blocks, include_top=include_top, do_p=do_p,
                        classes=classes, *args, **kwargs)


# def ResNet152_mc_dp_old(inputs, weights, blocks=None, include_top=True, do_p=0.3, classes=1000, *args, **kwargs):
#     """
#     Constructs a `keras.models.Model` according to the ResNet152 specifications with MC DropPath dropout.
#
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#
#     :param blocks: the network’s residual architecture
#
#     :param include_top: if true, includes classification layers
#
#     :param classes: number of classes to classify (include_top must be true)
#
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#
#     Usage:
#
#         >>> import keras_resnet.models
#
#         >>> shape, classes = (224, 224, 3), 1000
#
#         >>> x = keras.layers.Input(shape)
#
#         >>> model = keras_resnet.models.ResNet152(x, classes=classes)
#
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     if blocks is None:
#         blocks = [3, 8, 36, 3]
#     numerical_names = [False, True, True, False]
#
#     model = ResNet_do(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d_dp, include_top=include_top,
#                       net_type=NetType.mc_dp, do_p=do_p, classes=classes, *args, **kwargs)
#
#     donor_inputs = keras.layers.Input([*inputs.shape[1:-1], 3])
#     donor_model = donor_ResNet(donor_inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d,
#                                *args, **kwargs)
#     donor_model.load_weights(weights)
#
#     transfer_weights_from_donor_model(model, donor_model, inputs.shape[1:])
#     return model


def ResNet200(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet200 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet200(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 24, 36, 3]
    numerical_names = [False, True, True, False]

    return ResNet_do(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d_dp, include_top=include_top,
                     net_type=NetType.vanilla, do_p=0.3, classes=classes, *args, **kwargs)


def transfer_weights_from_donor_model(model, donor_model, input_shape):
    # donor_weights = donor_model.get_weights()
    # final_donor_weights = model.get_weights()[:1] + donor_weights[1:]
    # final_donor_weights[0] = np.concatenate((donor_weights[0], donor_weights[0][:, :, 0:input_shape[-1]-3, :]), axis=2)
    # model.set_weights(final_donor_weights)

    j = 1  # ignore input layers
    for i, l in enumerate(model.layers):
        if i < 1: # ignore input layers
            continue
        if j >= len(donor_model.layers):
            break
        d_l = donor_model.layers[j]
        if 'dropout' in l.name and 'dropout' not in d_l.name:
            continue
        j += 1
        if i == 2:
            new_w = tf.tile(d_l.weights[0], (1, 1, 2, 1))[:, :, :input_shape[-1], :]
            l.weights[0].assign(new_w)
        else:
            for (w, d_w) in zip(l.weights, d_l.weights):
                w.assign(d_w)
    assert j == len(donor_model.layers)

    # if weights != 'imagenet':
    #     print(f'Loading trained "{weights}" weights')
    #     f = h5py.File(weights, 'r')
    #     for i, l in enumerate(model.layers):
    #         l_ws = l.weights
    #         # print(len(f.keys()))
    #         # for k in f.keys():
    #         #    print(k)
    #         # input()
    #         d_ws = [f[l.name][l_w.name] for l_w in l_ws]
    #         if i == 1:
    #             new_w = np.concatenate((d_ws[0].value, l.weights[0].numpy()[..., 3:, :]), axis=-2)
    #             l.weights[0].assign(new_w)
    #             continue
    #         for (w, d_w) in zip(l.weights, d_ws):
    #             w.assign(d_w.value)
    del donor_model  # , donor_weights


import keras_resnet.layers

parameters = {
    "kernel_initializer": "he_normal"
}


def basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None):
    """
    A two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)
        y = keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    Usage:

        >>> import keras_resnet.blocks

        >>> bottleneck_2d_dp(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d_dp(filters, stage=0, block=0, kernel_size=3, do_p=0.3, net_type=NetType.mc, numerical_name=False, stride=None):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param do_p: float representing dropout rate

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    Usage:

        >>> import keras_resnet.blocks

        >>> bottleneck_2d_dp(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        if net_type == NetType.mc:
            y = keras.layers.Dropout(do_p)(y, training=True)
        elif net_type == NetType.mc_df:
            y = keras.layers.Dropout(do_p, noise_shape=(y.shape[0], 1, 1, y.shape[-1]))(y, training=True)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        if net_type == NetType.mc:
            y = keras.layers.Dropout(do_p)(y, training=True)
        elif net_type == NetType.mc_df:
            y = keras.layers.Dropout(do_p, noise_shape=(y.shape[0], 1, 1, y.shape[-1]))(y, training=True)
        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2c".format(stage_char, block_char))(y)
        if net_type == NetType.mc:
            y = keras.layers.Dropout(do_p)(y, training=True)
        elif net_type == NetType.mc_df:
            y = keras.layers.Dropout(do_p, noise_shape=(y.shape[0], 1, 1, y.shape[-1]))(y, training=True)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
            if net_type == NetType.mc:
                shortcut = keras.layers.Dropout(do_p)(shortcut, training=True)
            elif net_type == NetType.mc_df:
                shortcut = keras.layers.Dropout(do_p, noise_shape=(shortcut.shape[0], 1, 1, shortcut.shape[-1]))(shortcut, training=True)
        else:
            shortcut = x

        #keras.layers.Dropout() ???
        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

