
import subprocess
import sys

##### define experiments...
COMMANDS = [
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--experiment_root", "experiments/classification/single",
        "--encoder", "resnet18",
        "--input", "dem:dem.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--experiment_root", "experiments/classification/single",
        "--encoder", "resnet50",
        "--input", "dem:dem.tif"
    ],
    [
        "train_cls",
        "--config_path", "earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--experiment_root", "experiments/classification/single",
        "--encoder", "vit",
        "--input", "dem:dem.tif"
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