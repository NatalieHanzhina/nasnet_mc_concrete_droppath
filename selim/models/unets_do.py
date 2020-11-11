
from models.xception_padding1 import Xception
from models.xception_padding_do import Xception_do
from resnets import ResNet101, ResNet50
# from resnets_do import ResNet152_mc, ResNet152_mc_dp
from resnets_do import ResNet152, ResNet152_do
from resnetv2 import InceptionResNetV2Same
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
                    bn_scale=True, use_bias=True, dp_p=0.3):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    if net_type == net_type.mc:
        x = Dropout(dp_p)(x, training=True)
    elif net_type == net_type.mc_df:
        x = Dropout(dp_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
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
                 net_type=NetType.mc, dp_p=0.3):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    if net_type == net_type.mc:
        x = Dropout(dp_p)(x, training=True)
    elif net_type == net_type.mc_df:
        x = Dropout(dp_p, noise_shape=(x.shape[0], 1, 1, x.shape[-1]))(x, training=True)
    x = Activation(activation, name=name + '_relu')(x)
    return x


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


def decoder_block_no_bn_do(input, filters, skip, block_name, activation='relu', net_type=NetType.mc, dp_p=0.3):
    x = UpSampling2D()(input)
    x = conv_relu_do(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation,
                     net_type=net_type, dp_p=dp_p)
    # if net_type==NetType.mc_dp:
    #     Dropout() ???
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu_do(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation,
                     net_type=net_type, dp_p=dp_p)
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


def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x


def prediction_fpn_block_do(x, name, upsample=None, net_type=NetType.mc, dp_p=0.3):
    x = conv_relu_do(x, 128, 3, stride=1, name="prediction_" + name + "_1", net_type=net_type, dp_p=dp_p)
    x = conv_relu_do(x, 128, 3, stride=1, name="prediction_" + name + "_2", net_type=net_type, dp_p=dp_p)
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x


def resnet152_fpn(input_shape, channels=1, weights="imagenet", activation="softmax"):
    resnet_base = ResNet152(input_shape=input_shape, weights=weights, include_top=True)
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3b7_relu").output
    conv4 = resnet_base.get_layer("res4b35_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
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
    x = Conv2D(channels, (1, 1), name="mask", kernel_initializer="he_normal")(x)
    x = Activation(activation)(x)
    model = Model(resnet_base.input, x)
    return model


def resnet152_fpn_do(input_shape, net_type, channels=1, do_p=0.3, weights='imagenet', activation="softmax"):
    # if weights == 'imagenet':
    #     weights_file = download_resnet_imagenet("resnet152")
    # elif weights is None:
    #     weights_file = None
    # else:
    #     raise NotImplementedError('Only imagenet weights can be loaded')


    #print(net_type, '--------')
    resnet_base = ResNet152_do(input_shape=input_shape, include_top=True, net_type=net_type, do_p=do_p,
                               weights=weights)
    conv1 = resnet_base.get_layer("conv1_relu").output
    if net_type == NetType.mc_dp:
        conv1 = Dropout(do_p, noise_shape=[1 for _ in range(len(conv1.shape))])(conv1, training=True)
    conv2 = resnet_base.get_layer("res2c_relu").output
    if net_type == NetType.mc_dp:
        conv2 = Dropout(do_p, noise_shape=[1 for _ in range(len(conv2.shape))])(conv2, training=True)
    conv3 = resnet_base.get_layer("res3b7_relu").output
    if net_type == NetType.mc_dp:
        conv3 = Dropout(do_p, noise_shape=[1 for _ in range(len(conv3.shape))])(conv3, training=True)
    conv4 = resnet_base.get_layer("res4b35_relu").output
    if net_type == NetType.mc_dp:
        conv4 = Dropout(do_p, noise_shape=[1 for _ in range(len(conv4.shape))])(conv4, training=True)
    conv5 = resnet_base.get_layer("res5c_relu").output
    if net_type == NetType.mc_dp:
        conv5 = Dropout(do_p, noise_shape=[1 for _ in range(len(conv5.shape))])(conv5, training=True)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block_do(P5, "P5", (8, 8), net_type=net_type),
            prediction_fpn_block_do(P4, "P4", (4, 4), net_type=net_type),
            prediction_fpn_block_do(P3, "P3", (2, 2), net_type=net_type),
            prediction_fpn_block_do(P2, "P2", net_type=net_type),
        ]
    )
    x = conv_bn_relu_do(x, 256, 3, (1, 1), name="aggregation", net_type=net_type)
    x = decoder_block_no_bn_do(x, 128, conv1, 'up4', net_type=net_type)
    x = UpSampling2D()(x)
    x = conv_relu_do(x, 64, 3, (1, 1), name="up5_conv1", net_type=net_type)
    x = conv_relu_do(x, 64, 3, (1, 1), name="up5_conv2", net_type=net_type)
    x = Conv2D(channels, (1, 1), name="mask", kernel_initializer="he_normal")(x)
    x = Activation(activation)(x)
    model = Model(resnet_base.input, x)
    return model


def resnet152_fpn_mc(input_shape, channels=1, dp_p=0.3, weights='imagenet', activation="softmax"):
    return resnet152_fpn_do(input_shape, NetType.mc, channels, dp_p, weights, activation)


def resnet152_fpn_mc_df(input_shape, channels=1, dp_p=0.3, weights='imagenet', activation="softmax"):
    return resnet152_fpn_do(input_shape, NetType.mc_df, channels, dp_p, weights, activation)


def resnet152_fpn_mc_dp(input_shape, channels=1, dp_p=0.3, weights='imagenet', activation="softmax"):
    return resnet152_fpn_do(input_shape, NetType.mc_dp, channels, dp_p, weights, activation)


def resnet50_fpn(input_shape, channels=1, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet50(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3d_relu").output
    conv4 = resnet_base.get_layer("res4f_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
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
    model = Model(img_input, x)
    return model


def resnet101_fpn(input_shape, channels=1, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet101(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet101"))
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3b3_relu").output
    conv4 = resnet_base.get_layer("res4b22_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
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
    model = Model(img_input, x)
    return model


def xception_fpn(input_shape, channels=1, weights='imagenet', activation="sigmoid"):
    xception = Xception(input_shape=input_shape, weights=weights, include_top=False)
    conv1 = xception.get_layer("block1_conv2_act").output
    conv2 = xception.get_layer("block3_sepconv2_bn").output
    conv3 = xception.get_layer("block4_sepconv2_bn").output
    conv3 = Activation("relu")(conv3)
    conv4 = xception.get_layer("block13_sepconv2_bn").output
    conv4 = Activation("relu")(conv4)
    conv5 = xception.get_layer("block14_sepconv2_act").output

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
    model = Model(xception.input, x)
    return model


def xception_fpn_do(input_shape, net_type, channels=1, dp_p=0.3, weights='imagenet', activation="sigmoid"):
    xception_do = Xception_do(net_type=net_type, input_shape=input_shape, dp_p=dp_p, weights=weights, include_top=False)
    conv1 = xception_do.get_layer("block1_conv2_act").output
    conv2 = xception_do.get_layer("block3_sepconv2_bn").output
    conv3 = xception_do.get_layer("block4_sepconv2_bn").output
    conv3 = Activation("relu")(conv3)
    conv4 = xception_do.get_layer("block13_sepconv2_bn").output
    conv4 = Activation("relu")(conv4)
    conv5 = xception_do.get_layer("block14_sepconv2_act").output

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block_do(P5, "P5", (8, 8), dp_p=dp_p, net_type=net_type),
            prediction_fpn_block_do(P4, "P4", (4, 4), dp_p=dp_p, net_type=net_type),
            prediction_fpn_block_do(P3, "P3", (2, 2), dp_p=dp_p, net_type=net_type),
            prediction_fpn_block_do(P2, "P2", dp_p=dp_p, net_type=net_type),
        ]
    )
    x = conv_bn_relu_do(x, 256, 3, (1, 1), name="aggregation", net_type=net_type, dp_p=dp_p)
    x = decoder_block_no_bn_do(x, 128, conv1, 'up4', net_type=net_type, dp_p=dp_p)
    x = UpSampling2D()(x)
    x = conv_relu_do(x, 64, 3, (1, 1), name="up5_conv1", net_type=net_type, dp_p=dp_p)
    x = conv_relu_do(x, 64, 3, (1, 1), name="up5_conv2", net_type=net_type, dp_p=dp_p)
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(xception_do.input, x)
    return model


def xception_fpn_mc(input_shape, channels=1, dp_p=0.3, weights='imagenet', activation="sigmoid"):
    return xception_fpn_do(input_shape, NetType.mc, channels, dp_p, weights, activation)


def xception_fpn_mc_dp(input_shape, channels=1, dp_p=0.3, weights='imagenet', activation="sigmoid"):
    return xception_fpn_do(input_shape, NetType.mc_dp, channels, dp_p, weights, activation)


def densenet_fpn(input_shape, channels=1, activation="sigmoid"):
    densenet = DenseNet169(input_shape=input_shape, include_top=False)
    conv1 = densenet.get_layer("conv1/relu").output
    conv2 = densenet.get_layer("pool2_relu").output
    conv3 = densenet.get_layer("pool3_relu").output
    conv4 = densenet.get_layer("pool4_relu").output
    conv5 = densenet.get_layer("bn").output
    conv5 = Activation("relu", name="conv5_relu")(conv5)

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
    model = Model(densenet.input, x)
    return model


def inception_resnet_v2_fpn(input_shape, channels=1, activation="sigmoid"):
    inceresv2 = InceptionResNetV2Same(input_shape=input_shape, include_top=False)
    conv1, conv2, conv3, conv4, conv5 = inceresv2.output

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
    model = Model(inceresv2.input, x)
    return model


if __name__ == '__main__':
    resnet101_fpn((256, 256, 3)).summary()