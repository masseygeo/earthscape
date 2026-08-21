
import itertools
import subprocess
import sys

###################################################################
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

            if k == 'input':
                input_branches = (v if isinstance(v, (list,tuple)) else [v])
                for branch in input_branches:
                    exp += ["--input", str(branch)]
            else:
                exp += [f"--{k}", str(v)]
        
        # append to list of all experiments
        experiments.append(exp)

    return experiments
###################################################################

##### SET EXPERIMENT CONFIGS
# base train_cls CLI arguments...
# NOTE: subprocess expects each flag & value separate/tokenized
BASE = [
    sys.executable, "-m", "earthscape.cli.train_cls", 
    "--config_path", "configs_template.yml",
    "--mode", "train-test-cross",
    "--task", "classification",
    "--experiment_root", "experiments/sgmap-net/classification/variance_splits"
    ]

# experiment grid sweeps...
GRID = {
    'seed': [
        111,
        # 42, 
        # 90,
        # 256,
        # 128
    ],
    
    'model_name': [
        # 'resnet18', 
        # 'resnet50', 
        # 'vit',
        # 'swin',
        'sgmap-net', 
        # 'dofa', 
        # 'panopticon', 
        # 'copernicus-fm'
    ],

    'encoder_name': [
        'resnext50_32x4d', 
        'vit_b_16'
    ],

    'input': [
        ##### SINGLE MODALITIES
        # 'rgb:aerialr.tif,aerialg.tif,aerialb.tif', 
        # 'nir:aerialnir.tif', 
        # 'aerial:aerialr.tif,aerialg.tif,aerialb.tif,aerialnir.tif',
        'dem:dem.tif', 
        # 'nhd:nhd.tif', 'osm:osm.tif',
        # 'ep-5:ep_5x5.tif','ep-11:ep_11x11.tif','ep-21:ep_21x21.tif','ep-51:ep_51x51.tif','ep-101:ep_101x101.tif','ep-201:ep_201x201.tif',
        # 'plc-5:plc.tif','plc-10:plc_10.tif','plc-20:plc_20.tif','plc-50:plc_50.tif','plc-100:plc_100.tif','plc-200:plc_200.tif',
        # 'prc-5:prc.tif','prc-10:prc_10.tif','prc-20:prc_20.tif','prc-50:prc_50.tif','prc-100:prc_100.tif','prc-200:prc_200.tif',
        # 's-5:s.tif',
        # 's-10:s_10.tif',
        # 's-20:s_20.tif','s-50:s_50.tif','s-100:s_100.tif','s-200:s_200.tif',
        # 'sds-5:sds_5x5.tif','sds-11:sds_11x11.tif','sds-21:sds_21x21.tif','sds-51:sds_51x51.tif','sds-101:sds_101x101.tif','sds-201:sds_201x201.tif',


        ##### MULTI-SCALE
        # single branch (stacking & stacking+self-attention)...
        # 'ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif',
        # 'plc-ms:plc.tif,plc_10.tif,plc_20.tif,plc_50.tif,plc_100.tif,plc_200.tif',
        # 'prc-ms:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif',
        # 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'sds-ms:sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',

        # separate branches (concatenation & cross-attention + GAP)...
        # ['ep-5:ep_5x5.tif','ep-11:ep_11x11.tif','ep-21:ep_21x21.tif','ep-51:ep_51x51.tif','ep-101:ep_101x101.tif','ep-201:ep_201x201.tif'],
        # ['plc-5:plc.tif','plc-10:plc_10.tif','plc-20:plc_20.tif','plc-50:plc_50.tif','plc-100:plc_100.tif','plc-200:plc_200.tif'],
        # ['prc-5:prc.tif','prc-10:prc_10.tif','prc-20:prc_20.tif','prc-50:prc_50.tif','prc-100:prc_100.tif','prc-200:prc_200.tif'],
        # ['s-5:s.tif','s-10:s_10.tif','s-20:s_20.tif','s-50:s_50.tif','s-100:s_100.tif','s-200:s_200.tif'],
        # ['sds-5:sds_5x5.tif','sds-11:sds_11x11.tif','sds-21:sds_21x21.tif','sds-51:sds_51x51.tif','sds-101:sds_101x101.tif','sds-201:sds_201x201.tif'],


        ##### MULTIMODAL
        # single branch (stacking & stacking+self-attention)...
        # 'dem+s-ms:dem.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'dem+rgb:dem.tif,aerialr.tif,aerialg.tif,aerialb.tif',
        # 'ep-ms+prc-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif,prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif',
        # 'ep-ms+prc-ms+s-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif,prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'ep-ms+s-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'prc-ms+s-ms:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'ep-ms+s-ms+sds-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',
        # 'rgb+dem+ep-ms+s-ms+sds-ms: aerialr.tif,aerialg.tif,aerialb.tif,dem.tif,ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',
        # 'rgb+dem+nhd+osm:aerialr.tif,aerialg.tif,aerialb.tif,dem.tif,nhd.tif,osm.tif',
        # 'rgb+nhd+osm+s-ms: aerialr.tif,aerialg.tif,aerialb.tif,nhd.tif,osm.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',

        # separate branches (concatenation & cross-attention + GAP)...
        # ['dem:dem.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif'], 
        # ['dem:dem.tif', 'rgb:aerialr.tif,aerialg.tif,aerialb.tif'], 
        # ['ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif', 'prc-ms:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif'], 
        # ['ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif', 'prc-ms:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif'], 
        # ['ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif'], 
        # ['prc-ms:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif'], 
        # ['ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif', 'sds-ms:sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif'], 
        # ['rgb:aerialr.tif,aerialg.tif,aerialb.tif', 'dem:dem.tif', 'ep-ms:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif', 'sds-ms:sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif'], 
        # ['rgb:aerialr.tif,aerialg.tif,aerialb.tif', 'dem:dem.tif', 'nhd:nhd.tif', 'osm:osm.tif',], 
        # ['rgb:aerialr.tif,aerialg.tif,aerialb.tif', 'nhd:nhd.tif', 'osm:osm.tif', 's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',], 
    ],
        
    'embedding_fusion': ['none'],

    'split_dir': [
        # 'splits/esv1p1_splits', 
        'splits/split_seed_42', 
        'splits/split_seed_90', 
        'splits/split_seed_128', 
        'splits/split_seed_256', 
        ],

    'batch_size': [16],

    # 'lr': [1e-3, 1e-4, 1e-5],

    }



##### RUN EXPERIMENTS
# list of lists of experiments
EXPERIMENTS = make_experiments(BASE, GRID)

# iterate & run experiments...
for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"\nExperiment {i}/{len(EXPERIMENTS)}")
    print(" ".join(exp))
    result = subprocess.run(exp, check=True)

    if result.returncode != 0:
        print("Experiment failed! Stopping...")
        sys.exit(result.returncode)