
import itertools
import subprocess
import sys


###################################################################
def make_experiments(base, grid, model_encoders):
    experiments = []
    grid_keys = list(grid.keys())
    for model_name, encoder_names in model_encoders.items():
        for encoder_name in encoder_names:
            for values in itertools.product(*(grid[k] for k in grid_keys)):
                exp = base.copy()

                # fixed pair
                exp += ["--model_name", model_name]
                exp += ["--encoder_name", encoder_name]

                # normal sweep params
                for k, v in zip(grid_keys, values):
                    exp += [f"--{k}", str(v)]

                experiments.append(exp)

    return experiments
###################################################################


##### set up experiment arguments...
# required CLI arguments (non-changing)...
BASE = [
    sys.executable, "-m", "earthscape.cli.train_seg", 
    "--config_path", "earthscape/configs_template.yml",
    "--mode", "train-test-cross",
    "--experiment_root", "experiments/baselines/segmentation"
    ]

# models & specific backbones...
MODEL_ENCODERS = {
    "unet": ["resnet18"],
    "deeplabv3p": ["resnet50"],
    "segformer": ["mit_b0"],
    }

# input hyperparameter sweeps...
GRID = {
    'input': [

        ##### BASELINE EXPERIMENTS
        'dem:dem.tif', 
        'rgb:aerialr.tif,aerialg.tif,aerialb.tif', 
        'dem+rgb:dem.tif,aerialr.tif,aerialg.tif,aerialb.tif',
        's5:s.tif','s10:s_10.tif','s20:s_20.tif','s50:s_50.tif','s100:s_100.tif','s200:s_200.tif',
        's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',

        ]
    }


##### run experiments...
EXPERIMENTS = make_experiments(BASE, GRID, MODEL_ENCODERS)
for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"\nExperiment {i}/{len(EXPERIMENTS)}")
    print(" ".join(exp))
    result = subprocess.run(exp, check=True)
    if result.returncode != 0:
        print("Experiment failed! Stopping...")
        sys.exit(result.returncode)

