
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
    "--config_path", "earthscape/configs_template.yml",
    "--mode", "train-test-cross",
    "--experiment_root", "experiments/baselines/classification"
    ]

# experiment grid sweeps...
GRID = {
    'model_name': [
        'resnet18', 
        'resnet50', 
        'vit',
        'swin'
        ],

    'input': [

        ##### BASELINE EXPERIMENTS
        'dem:dem.tif',
        'rgb:aerialr.tif,aerialg.tif,aerialb.tif', 
        'dem+rgb:dem.tif,aerialr.tif,aerialg.tif,aerialb.tif',
        's5:s.tif','s10:s_10.tif','s20:s_20.tif','s50:s_50.tif','s100:s_100.tif','s200:s_200.tif',
        's-ms:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',


        ##### SINGLE MODALITIES
        ##### modality reliability under geographc domain shift)...
        # NOTE: Which modalities are reliable predictors under geographic distribution shift?
        # 'rgb:aerialr.tif,aerialg.tif,aerialb.tif', 
        # 'nir:aerialnir.tif', 
        # 'aerial:aerialr.tif,aerialg.tif,aerialb.tif,aerialnir.tif', 
        # 'dem:dem.tif', 
        # 'nhd:nhd.tif', 'osm:osm.tif',
        # 'ep5:ep_5x5.tif','ep11:ep_11x11.tif','ep21:ep_21x21.tif','ep51:ep_51x51.tif','ep101:ep_101x101.tif','ep201:ep_201x201.tif',
        # 'plc5:plc.tif','plc10:plc_10.tif','plc20:plc_20.tif','plc50:plc_50.tif','plc100:plc_100.tif','plc200:plc_200.tif',
        # 'prc5:prc.tif','prc10:prc_10.tif','prc20:prc_20.tif','prc50:prc_50.tif','prc100:prc_100.tif','prc200:prc_200.tif',
        # 's5:s.tif','s10:s_10.tif','s20:s_20.tif','s50:s_50.tif','s100:s_100.tif','s200:s_200.tif',
        # 'sds5:sds_5x5.tif','sds11:sds_11x11.tif','sds21:sds_21x21.tif','sds51:sds_51x51.tif','sds101:sds_101x101.tif','sds201:sds_201x201.tif',


        ##### MULTI-SCALE
        ##### effect of scale & multiscale representations...
        # NOTE: Do multiscale terrain representations improve cross-domain robustness compared to single-scale inputs?
        # 'ep:ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif',
        # 'plc:plc.tif,plc_10.tif,plc_20.tif,plc_50.tif,plc_100.tif,plc_200.tif',
        # 'prc:prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif',
        # 's:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'sds:sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',


        ##### MULTIMODAL
        ##### raw sensor combinations...

        # 'dem_rgb:dem.tif,aerialr.tif,aerialg.tif,aerialb.tif',
        # 'dem_aerial:dem.tif,aerialr.tif,aerialg.tif,aerialb.tif,aerialnir.tif',

        ##### complementarity of weak modalities...
        # NOTE: Do weak standalone modalities provide complementary information when combined with a robust representation?
        # define robust core representation (based on single & multiscale results)
        # 1. core baselines
        # 2. context add-ons (osm & nhd)
        # 3. terrain family add ons (prc, plc, etc.)
        # 4. 3+ modality tests

        # 'dem_nhd:dem.tif,nhd.tif',
        # 'dem_osm:dem.tif,osm.tif',
        

        ##### negative transfer from raw modalities...
        # NOTE: Can additional modalities improve in-domain performance while degrading cross-domain performance?
        # use same core as above
        # 1. add raw modalities to core (RGB, DEM, RGB+DEM)
        # 2. possibly best complementary combo (core + complement + raw sensor(s))

        # 'dem_ep:dem.tif,ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif',
        # 'dem_plc:dem.tif,plc.tif,plc_10.tif,plc_20.tif,plc_50.tif,plc_100.tif,plc_200.tif',
        # 'dem_prc:dem.tif,prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif',
        # 'dem_s:dem.tif,s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif',
        # 'dem_sds:dem.tif,sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',

        ##### class-specific modality specialization
        # NOTE: Do modalities specialize for different classes, and can this inform multimodal stacks?
        # 1. class-informed stack (CIS)
        # 2. CIS + complements
        # 3. CIS + raw sensors (RGB, DEM, RGB+DEM)


        # 's_rgb:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,aerialr.tif,aerialg.tif,aerialb.tif',
        # 's_aerial:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,aerialr.tif,aerialg.tif,aerialb.tif,aerialnir.tif',
        # 's_nhd:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,nhd.tif',
        # 's_osm:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,osm.tif',
        # 's_ep:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,ep_5x5.tif,ep_11x11.tif,ep_21x21.tif,ep_51x51.tif,ep_101x101.tif,ep_201x201.tif',
        # 's_plc:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,plc.tif,plc_10.tif,plc_20.tif,plc_50.tif,plc_100.tif,plc_200.tif',
        # 's_prc:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,prc.tif,prc_10.tif,prc_20.tif,prc_50.tif,prc_100.tif,prc_200.tif'
        # 's_sds:s.tif,s_10.tif,s_20.tif,s_50.tif,s_100.tif,s_200.tif,sds_5x5.tif,sds_11x11.tif,sds_21x21.tif,sds_51x51.tif,sds_101x101.tif,sds_201x201.tif',
        ]
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