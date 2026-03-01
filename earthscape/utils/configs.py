
import yaml



def config_load(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg



def config_update(cfg, args):

    def _opt(name, default=None):
        return getattr(args, name, default)

    if _opt("experiment_root") is not None:
        cfg['experiment']['root'] = _opt("experiment_root")
    if _opt("seed") is not None:
        cfg['experiment']['seed'] = _opt("seed")
    if _opt("compile") is not None:
        cfg['model']['compile'] = args.compile == 'true'
    if _opt("encoder") is not None:
        cfg['model']['encoder'] = _opt("encoder")
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
    out = {}
    for input_features in arg_inputs or []:
        name, channels = input_features.split(':')
        channels = [c.strip() for c in channels.split(',') if c.strip()]
        out[name] = {'channels': channels}
    return out



# def config_save(cfg, output_path):
#     cfg_copy = cfg.copy()
#     cfg_copy["experiment"]["config_path"] = str(Path(config_path).resolve())
#     out_path = run_dir / "config_used.yml"
#     with open(out_path, "w") as f:
#         yaml.safe_dump(cfg_copy, f)
