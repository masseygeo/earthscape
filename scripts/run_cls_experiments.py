
import subprocess
import sys

##### define experiments...
COMMANDS = [
    [
        "train_cls",
        "--config_path", "./earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--experiment_root", "experiments/test",
        "--encoder", "resnet18",
        "--input", "dem:dem.tif"
    ],
    [
        "train_cls",
        "--config_path", "./earthscape/configs_template.yml",
        "--mode", "train-test-cross",
        "--experiment_root", "experiments/test",
        "--encoder", "resnet18",
        "--input", "slope:s.tif"
    ],
]


##### iterate & run experiments...
for cmd in COMMANDS:
    print("\n>>> Running:\n", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Experiment failed! Stopping...")
        sys.exit(result.returncode)

print("\nAll experiments finished successfully.")