import yaml




def config_load(path):
    """
    Load a configuration file from disk.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration data.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg



def config_update(cfg, args):
    """
    Update a configuration dictionary (from earthscape experiment config.yml file) 
    with values from an argument namespace.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary to be updated in place.
    args : object
        Object exposing attributes corresponding to configurable parameters.
        Missing attributes or attributes set to None are ignored.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """

    def _opt(name, default=None):
        """Retrieve an attribute from the argument namespace with a default fallback."""
        return getattr(args, name, default)

    if _opt("experiment_root") is not None:
        cfg['experiment']['root'] = _opt("experiment_root")
    if _opt("seed") is not None:
        cfg['experiment']['seed'] = _opt("seed")
    if _opt("compile") is not None:
        cfg['model']['compile'] = args.compile == 'true'
    if _opt("model_name") is not None:
        cfg['model']['model_name'] = _opt("model_name")
    if _opt("encoder_name") is not None:
        cfg['model']['encoder_name'] = _opt("encoder_name")
    if _opt("area_threshold") is not None:
        cfg['labels']['area_threshold'] = _opt("area_threshold")
    if _opt("batch_size") is not None:
        cfg['dataloader']['base']['batch_size'] = _opt("batch_size")
    if _opt("lr") is not None:
        cfg['optimizer']['params']['lr'] = _opt("lr")
    if _opt("weight_decay") is not None:
        cfg['optimizer']['params']['weight_decay'] = _opt("weight_decay")
    if _opt("pos_weight") is not None:
        cfg['loss']['params']['pos_weight'] = _opt("pos_weight")
    if _opt("gamma") is not None:
        cfg['loss']['params']['gamma'] = _opt("gamma")
    if _opt("alpha") is not None:
        cfg['loss']['params']['alpha'] = _opt("alpha")
    if _opt("reduction") is not None:
        cfg['loss']['params']['reduction'] = _opt("reduction")
    if _opt("input"):
        cfg['data']['input'] = _parse_inputs(_opt("input"))
    if _opt("patience") is not None:
        cfg['early_stop']['patience'] = _opt("patience")
    if _opt("min_delta") is not None:
        cfg['early_stop']['min_delta'] = _opt("min_delta")
    if _opt("warmup_epochs") is not None:
        cfg['early_stop']['warmup_epochs'] = _opt("warmup_epochs")
    return cfg



def _parse_inputs(arg_inputs):
    """
    Parse input feature specifications into a structured mapping.

    Parameters
    ----------
    arg_inputs : iterable of str or None
        Input feature specifications of the form "name:channel1,channel2,...".
        If None, no inputs are parsed.

    Returns
    -------
    dict
        Mapping from input names to dictionaries with a "channels" key
        containing a list of channel names.
    """
    out = {}
    for input_features in arg_inputs or []:
        name, channels = input_features.split(':')
        channels = [c.strip() for c in channels.split(',') if c.strip()]
        out[name] = {'channels': channels}
    return out
