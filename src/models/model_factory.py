from models.unets_do import nasnet_sch_dp_cls, nasnet_fpn_mc_sch_dp,\
    nasnet_cls, nasnet_cdp_cls, nasnet_do_cls, nasnet_df_cls


def make_model(network, input_shape, pretrained_weights, do_p=0.3, **kwargs):

    if network == 'nasnet_cdp_cls': #our
        return nasnet_cdp_cls(input_shape, weights=pretrained_weights, activation="softmax",
                              **kwargs)
    elif network == 'nasnet_do_cls':
        return nasnet_do_cls(input_shape, do_p=do_p, weights=pretrained_weights, activation="softmax",
                              **kwargs)
    elif network == 'nasnet_df_cls':
        return nasnet_df_cls(input_shape, do_p=do_p, weights=pretrained_weights, activation="softmax",
                              **kwargs)
    elif network == 'nasnet_sch_dp_cls': #our
        return nasnet_sch_dp_cls(input_shape, do_p=do_p, weights=pretrained_weights, activation="softmax",
                              **kwargs)
    elif network == 'nasnet_cls':
        return nasnet_cls(input_shape, do_p=do_p, weights=pretrained_weights, activation="softmax",
                              **kwargs)

    elif network == 'nasnet_mc_dp_test':     #  does not support scheduling dropout during training. Used for shceduled mc inferences of sheduled-trained models with droppath
        return nasnet_fpn_mc_sch_dp(input_shape, channels=2, do_p=do_p, weights=pretrained_weights,
                                    activation="sigmoid", **kwargs)
    else:
        raise ValueError('unknown network ' + network)
