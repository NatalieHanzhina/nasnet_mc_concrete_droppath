from models.nasnet_do import NASNet_large_do

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dropout, UpSampling2D, Conv2D, BatchNormalization, Activation, concatenate, Add
from tensorflow.keras.utils import get_file

from . import NetType

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


def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn_relu_do(input, num_channel, kernel_size, stride, name, padding='same', net_type=NetType.mc, bn_axis=-1, bn_momentum=0.99,
                    bn_scale=True, use_bias=True, do_p=0.3):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    if net_type == net_type.mc:
        x = Dropout(do_p)(x, training=True)
    elif net_type == net_type.mc_df:
        x = Dropout(do_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99, bn_scale=True,
            use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    return x


def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = Activation(activation, name=name + '_relu')(x)
    return x


def conv_relu_do(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu',
                 net_type=NetType.mc, do_p=0.3):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    if net_type == net_type.mc:
        x = Dropout(do_p)(x, training=True)
    elif net_type == net_type.mc_df:
        x = Dropout(do_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = Activation(activation, name=name + '_relu')(x)
    return x


def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal")(C5)
    P5_upsampled = UpSampling2D(name='P5_upsampled')(P5)

    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                kernel_initializer="he_normal")(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = UpSampling2D(name='P4_upsampled')(P4)

    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                kernel_initializer="he_normal")(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = UpSampling2D(name='P3_upsampled')(P3)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = Add(name='P1_merged')([P2_upsampled, P1])
    P1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal")(P1)

    return P1, P2, P3, P4, P5


def decoder_block(input, filters, skip, block_name):
    x = UpSampling2D()(input)
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1')
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2')
    return x


def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x


def decoder_block_no_bn_do(input, filters, skip, block_name, activation='relu', net_type=NetType.mc, do_p=0.3):
    x = UpSampling2D()(input)
    x = conv_relu_do(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation,
                     net_type=net_type, do_p=do_p)
    # if net_type==NetType.mc_dp:
    #     Dropout() ???
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu_do(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation,
                     net_type=net_type, do_p=do_p)
    return x

def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x


def prediction_fpn_block_do(x, name, upsample=None, net_type=NetType.mc, do_p=0.3):
    x = conv_relu_do(x, 128, 3, stride=1, name="prediction_" + name + "_1", net_type=net_type, do_p=do_p)
    x = conv_relu_do(x, 128, 3, stride=1, name="prediction_" + name + "_2", net_type=net_type, do_p=do_p)
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x

def nasnet_cls(input_shape, do_p=0, resize_size=32, total_training_steps=None, weights='imagenet', activation="softmax",
               classes=13):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    nasnet = NASNet_large_do(input_shape=input_shape, net_type=NetType.vanilla, do_p=do_p, include_top=False,
                             total_training_steps=total_training_steps, activation=activation, weights=weights, classes=classes)
    return nasnet

def nasnet_cdp_cls(input_shape, do_p=0.3, resize_size=32, total_training_steps=None,
                         weights='imagenet', activation="softmax", classes=13):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return NASNet_large_do(input_shape=input_shape, net_type=NetType.cdp, do_p=do_p, total_training_steps=total_training_steps,
                           weights=weights, activation=activation, classes=classes, include_top=False)

def nasnet_sch_dp_cls(input_shape, do_p=0.3, resize_size=32, total_training_steps=None, # IS NOT IMPLEMENTED
                         weights='imagenet', activation="softmax", classes=13):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return NASNet_large_do(input_shape=input_shape, net_type=NetType.mc_dp, do_p=do_p, total_training_steps=total_training_steps,
                           weights=weights, activation=activation, classes=classes, include_top=False)

def nasnet_do_cls(input_shape, do_p=0.3, resize_size=32, total_training_steps=None,
                         weights='imagenet', activation="softmax", classes=13):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return NASNet_large_do(input_shape=input_shape, net_type=NetType.mc, do_p=do_p, total_training_steps=total_training_steps,
                           weights=weights, activation=activation, classes=classes, include_top=False)

def nasnet_df_cls(input_shape, do_p=0.3, resize_size=32, total_training_steps=None,
                         weights='imagenet', activation="softmax", classes=13):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return NASNet_large_do(net_type=NetType.mc_df, do_p=do_p, total_training_steps=total_training_steps, weights=weights,
                           input_shape=input_shape, activation=activation, classes=classes, include_top=False)



def nasnet_fpn_do(input_shape, net_type, channels=1, do_p=0.3, total_training_steps=None, weights='imagenet', activation="softmax"):
    nasnet = NASNet_large_do(input_shape=input_shape, net_type=net_type, do_p=do_p, include_top=False,
                             total_training_steps=total_training_steps, weights=weights)
    conv1 = nasnet.get_layer("activation").output  # ("stem_bn1").output
    conv2 = nasnet.get_layer("reduction_concat_stem_1").output
    conv3 = nasnet.get_layer("activation_134").output  # ("normal_concat_5").output
    conv4 = nasnet.get_layer("activation_252").output  # ("normal_concat_12").output  # shape: (batch_size, 16, 16, channels)
    conv5 = nasnet.get_layer("normal_concat_18").output  # ("normal_concat_18").output  # shape: (batch_size, 8, 8, channels)

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(nasnet.input, x)
    return model


def nasnet_cdp_fpn(input_shape, channels=1, do_p=None, resize_size=None, total_training_steps=None, weights='imagenet',
                   activation="sigmoid"):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return nasnet_fpn_do(input_shape, NetType.cdp, channels, weights=weights, activation=activation)


def nasnet_do_fpn(input_shape, channels=1, do_p=0.3, resize_size=None, total_training_steps=None, weights='imagenet',
                  activation="sigmoid"):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return nasnet_fpn_do(input_shape, NetType.mc, channels, do_p, weights=weights, activation=activation)


def nasnet_df_fpn(input_shape, channels=1, do_p=0.3, resize_size=None, total_training_steps=None, weights='imagenet',
                  activation="sigmoid"):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return nasnet_fpn_do(input_shape, NetType.mc_df, channels, do_p, weights=weights, activation=activation)

#vanilla nasnet
def nasnet_scd_fpn(input_shape, channels=1, do_p=0.3, resize_size=None, total_training_steps=None,
                   weights='imagenet', activation="sigmoid"):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return nasnet_fpn_do(input_shape, NetType.sdp, channels, do_p, total_training_steps, weights, activation)


def nasnet_fpn_mc_sch_dp(input_shape, channels=1, do_p=0.3, resize_size=None, total_training_steps=None,
                         weights='imagenet', activation="sigmoid"):
    if resize_size is not None:
        input_shape = (*((resize_size, resize_size) if isinstance(resize_size, int) else resize_size), input_shape[2])
    return nasnet_fpn_do(input_shape, NetType.mc_dp, channels, do_p, total_training_steps, weights, activation)  #TODO: refactor NetType to mc_sch_dp

