from models.unets_do import densenet_fpn, densenet_fpn_mc, densenet_fpn_mc_df, inception_resnet_v2_fpn, inception_resnet_v2_fpn_mc_dp, \
    inception_resnet_v2_fpn_sch_do, nasnet_cdp_fpn, nasnet_do_fpn, nasnet_df_fpn, nasnet_scd_fpn, \
    nasnet_fpn_mc_sch_dp, resnet152_fpn, resnet152_fpn_mc, resnet152_fpn_mc_df, resnet152_fpn_mc_dp, resnet101_fpn,\
    resnet50_fpn, resnext50_fpn, xception_fpn, xception_fpn_mc, xception_fpn_mc_df, xception_fpn_mc_dp


def make_model(network, input_shape, pretrained_weights, do_p=0.3, **kwargs):
    if network == 'densenet169':
        return densenet_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'densenet169_softmax':
        return densenet_fpn(input_shape, channels=3, activation="softmax")
    elif network == 'densenet169_mc':
        return densenet_fpn_mc(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'densenet169_mc_df':
        return densenet_fpn_mc_df(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_cdp':
        return nasnet_cdp_fpn(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid",
                              **kwargs)
    elif network == 'nasnet_do':
        return nasnet_do_fpn(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid",
                             **kwargs)
    elif network == 'nasnet_df':
        return nasnet_df_fpn(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid",
                             **kwargs)
    elif network == 'nasnet_sch_dp':
        return nasnet_scd_fpn(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid",
                              **kwargs)
    elif network == 'nasnet_mc_dp_test':     #  does not support scheduling dropout during training. Used for shceduled mc inferences of sheduled-trained models with droppath
        return nasnet_fpn_mc_sch_dp(input_shape, channels=2, do_p=do_p, weights=pretrained_weights,
                                    activation="sigmoid", **kwargs)
    if network == 'resnet101_softmax':
        return resnet101_fpn(input_shape, channels=3, activation="softmax")
    elif network == 'resnet152_2':
        return resnet152_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet152_2_mc':
        return resnet152_fpn_mc(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'resnet152_2_mc_df':
        return resnet152_fpn_mc_df(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'resnet152_2_mc_dp':
        return resnet152_fpn_mc_dp(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'resnet101_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnetv2':
        return inception_resnet_v2_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'resnetv2_mc_dp':
        return inception_resnet_v2_fpn_mc_dp(input_shape, channels=2, do_p=do_p, weights=pretrained_weights,
                                             activation="sigmoid")
    elif network == 'resnetv2_sch_dp':
        return inception_resnet_v2_fpn_sch_do(input_shape, channels=2, do_p=do_p, weights=pretrained_weights,
                                              activation="sigmoid", **kwargs)
    elif network == 'resnetv2_3':
        return inception_resnet_v2_fpn(input_shape, channels=3, activation="sigmoid")
    elif network == 'resnet101_unet_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnext50':
        return resnext50_fpn(input_shape, channels=2, cardinality=32, activation="sigmoid")
    elif network == 'xception':
        return xception_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'xception_mc':
        return xception_fpn_mc(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'xception_mc_df':
        return xception_fpn_mc_df(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    elif network == 'xception_mc_dp':
        return xception_fpn_mc_dp(input_shape, channels=2, do_p=do_p, weights=pretrained_weights, activation="sigmoid")
    else:
        raise ValueError('unknown network ' + network)
