import tensorflow as tf

from models.model_factory import make_model


def cross_valid_ensemble():
    pass


def random_init_ensemble(network_args, networks_count, network_seeds=None):
    if network_seeds is None:
        network_seeds = list(range(0, networks_count * 300, 300))
    ensemble = []
    for ntw_seed in network_seeds:
        tf.random.set_seed(ntw_seed)
        ensemble.append(make_model(network_args))
    return ensemble