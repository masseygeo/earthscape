
import itertools
import subprocess
import sys


# base train_cls CLI arguments...
# NOTE: subprocess expects each flag & value separate/tokenized
BASE = [
    sys.executable, "-m", "earthscape.cli.train_cls", 
    "--config_path", "earthscape/configs_template.yml",
    "--mode", "train-test-cross"
    ]


# experiment grid sweeps...
GRID = {
    'encoder': ['resnet18', 'resnet50', 'vit'],
    'lr': [1e-3, 3e-4, 1e-4],
    'batch_size': [32, 64, 128],
    'input': ['dem:dem.tif', 'slope:s.tif']
    }


def make_experiments(base, grid):
    """function to build tokenized experiment CLI lists"""
    
    # initialize list to hold experiments
    experiments = []

    # experiment override flag name(s)
    keys = list(grid.keys())

    # iterate over cartesion product of grid values (full sweep)
    for values in itertools.product(*(grid[k] for k in keys)):

        # copy BASE to not mutate
        exp = base.copy()

        # build flag and argument as separate tokens & build experiment list
        for k, v in zip(keys, values):
            exp += [f"--{k}", str(v)]
        
        # append to list of all experiments
        experiments.append(exp)

    return experiments


# list of lists of experiments
EXPERIMENTS = make_experiments(BASE, GRID)


##### iterate & run experiments...
for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f">>> Experiment {i}/{len(EXPERIMENTS)}...")
    print(" ".join(exp))
    result = subprocess.run(exp, check=True)

    if result.returncode != 0:
        print("Experiment failed! Stopping...")
        sys.exit(result.returncode)

