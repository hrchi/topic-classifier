import argparse
import yaml
from train import train_model
from eval import evaluate_model
from utils.seed import set_seed

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if config["mode"] == "train":
        print("🔧 Running training...")
        train_model(config)
    elif config["mode"] == "eval":
        print("🔧 Running evaluation...")
        evaluate_model(config)
    else:
        print("❌ Unknown mode:", config["mode"])

if __name__ == "__main__":
    main()
