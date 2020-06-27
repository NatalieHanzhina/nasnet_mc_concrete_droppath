from models.unets import resnet152_fpn, resnet152_fpn_mc, resnet152_fpn_mc_dp, resnet101_fpn, resnet50_fpn, xception_fpn, xception_fpn_mc, xception_fpn_mc_dp, densenet_fpn, inception_resnet_v2_fpn


def make_model(network, input_shape, pretrained_weights, mc_dp=0.3):
    if network == 'resnet101_softmax':
        return resnet101_fpn(input_shape,channels=3, activation="softmax")
    elif network == 'resnet152_2':
        return resnet152_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet152_2_mc':
        return resnet152_fpn_mc(input_shape, channels=2, mc_dp=mc_dp, activation="sigmoid")
    elif network == 'resnet152_2_mc_dp':
        return resnet152_fpn_mc_dp(input_shape, channels=2, mc_dp=mc_dp, activation="sigmoid")
    elif network == 'resnet101_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnetv2':
        return inception_resnet_v2_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnetv2_3':
        return inception_resnet_v2_fpn(input_shape, channels=3, activation="sigmoid")
    elif network == 'densenet169':
        return densenet_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'densenet169_softmax':
        return densenet_fpn(input_shape, channels=3, activation="softmax")
    elif network == 'resnet101_unet_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'xception_fpn':
        return xception_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'xception_fpn_mc':
        return xception_fpn_mc(input_shape, channels=2, dp_p=mc_dp, weights=pretrained_weights, activation="sigmoid")
    elif network == 'xception_fpn_mc_dp':
        return xception_fpn_mc_dp(input_shape, channels=2, dp_p=mc_dp, weights=pretrained_weights, activation="sigmoid")
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=2, activation="sigmoid")
    else:
        raise ValueError('unknown network ' + network)
