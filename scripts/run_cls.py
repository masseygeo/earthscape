
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
    'encoder': ['resnet18'],
    'input': [
        # single modalities...
        # 'rgb:aerialr.tif,aerialg.tif,aerialb.tif', 'nir:aerialnir.tif', 'aerial:aerialr.tif,aerialg.tif,aerialb.tif,aerialnir.tif', 
        # 'dem:dem.tif', 'nhd:nhd.tif', 'osm:osm.tif',
        # 'ep5:ep_5x5.tif','ep11:ep_11x11.tif','ep21:ep_21x21.tif','ep51:ep_51x51.tif','ep101:ep_101x101.tif','ep201:ep_201x201.tif',
        # 'plc5:plc.tif','plc10:plc_10.tif','plc20:plc_20.tif','plc50:plc_50.tif','plc100:plc_100.tif','plc200:plc_200.tif',
        # 'prc5:prc.tif','prc10:prc_10.tif','prc20:prc_20.tif','prc50:prc_50.tif','prc100:prc_100.tif','prc200:prc_200.tif',
        # 's5:s.tif','s10:s_10.tif','s20:s_20.tif','s50:s_50.tif','s100:s_100.tif','s200:s_200.tif',
        # 'sds5:sds_5x5.tif','sds11:sds_11x11.tif','sds21:sds_21x21.tif','sds51:sds_51x51.tif','sds101:sds_101x101.tif','sds201:sds_201x201.tif',

        # multi-scale terrain features...
        'ep:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif',
        'plc:plc.tif,plc_10.tif,plc_20.tif,plc_50.tif,plc_100.tif,plc_200.tif',
        'prc:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif',
        's:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        'sds:sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',

        # multimodal inputs...
        
        ]
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

