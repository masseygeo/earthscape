
import subprocess
import sys

##### define experiments...
COMMANDS = [
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "dem:dem.tif"
    ],

    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "rgb:aerialr.tif,aerialg.tif,aerialb.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "nir:aerialnir.tif"
    ],

    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "nhd:nhd.tif"
    ],

    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "osm:osm.tif"
    ],

    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep5:ep_5x5.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep11:ep_11x11.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep21:ep_21x21.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep51:ep_51x51.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep101:ep_101x101.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "ep201:ep_201x201.tif"
    ],

    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc5:plc.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc10:plc_10.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc20:plc_20.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc50:plc_50.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc100:plc_100.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "plc200:plc_200.tif"
    ],


    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc5:prc.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc10:prc_10.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc20:prc_20.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc50:prc_50.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc100:prc_100.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "prc200:prc_200.tif"
    ],


    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s5:s.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s10:s_10.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s20:s_20.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s50:s_50.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s100:s_100.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "s200:s_200.tif"
    ],


    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds5:sds_5x5.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds11:sds_11x11.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds21:sds_21x21.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds51:sds_51x51.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds101:sds_101x101.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--encoder", "resnet18",
        "--batch_size", "32",
        "--patience", "10",
        "--input", "sds201:sds_201x201.tif"
    ],
]


##### iterate & run experiments...
for cmd in COMMANDS:
    print("\n>>> Running:\n", " ".join(cmd))
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print("Experiment failed! Stopping...")
        sys.exit(result.returncode)

print("\nAll experiments finished successfully.\n")