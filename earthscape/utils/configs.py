
import yaml



def config_load(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg



def config_update(cfg, args):
    if args.seed is not None:
        cfg['experiment']['seed'] = args.seed
    if args.compile is not None:
        cfg['model']['compile'] = args.compile == 'true'
    if args.encoder is not None:
        cfg['model']['encoder'] = args.encoder
    if args.area_threshold is not None:
        cfg['labels']['area_threshold'] = args.area_threshold
    if args.batch_size is not None:
        cfg['dataloader']['base']['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['optimizer']['params']['lr'] = args.lr
    if args.weight_decay is not None:
        cfg['optimizer']['params']['weight_decay'] = args.weight_decay
    if args.pos_weight is not None:
        cfg['loss']['params']['pos_weight'] = args.pos_weight
    if args.gamma is not None:
        cfg['loss']['params']['gamma'] = args.gamma
    if args.alpha is not None:
        cfg['loss']['params']['alpha'] = args.alpha
    if args.reduction is not None:
        cfg['loss']['params']['reduction'] = args.reduction
    
    if args.input:
        cfg['data']['input'] = _parse_inputs(args.input)

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
