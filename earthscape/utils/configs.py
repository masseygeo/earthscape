
import yaml
from pathlib import Path




def config_load(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg




def config_save(cfg, run_dir, config_path):
    cfg_copy = cfg.copy()
    cfg_copy["experiment"]["config_path"] = str(Path(config_path).resolve())
    out_path = run_dir / "config_used.yml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg_copy, f)
