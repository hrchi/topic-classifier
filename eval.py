import os

import torch

from model.dnn import DNNClassifier
from utils.data_utils import prepare_eval_loader
from utils.eval_utils import evaluate


def evaluate_model(config):
    print("🔍 Running final evaluation on test set...")

    test_loader, vocab = prepare_eval_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model_path = os.path.join(config["misc"]["save_dir"], config["misc"]["model_file"])
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, f1 = evaluate(model, test_loader, device)
    print(f"✅ Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
