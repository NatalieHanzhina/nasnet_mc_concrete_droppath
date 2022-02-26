from models.ensemble import cross_valid_ensemble


def make_ensemble(ensemble_type, network_args, **kwargs):
    if ensemble_type == 'cross_validation':
        return cross_valid_ensemble(network_args, **kwargs)
    elif ensemble_type == 'random_init':
        return random_init_ensemble(network_args, **kwargs)
    elif ensemble_type == 'combined':
        pass
