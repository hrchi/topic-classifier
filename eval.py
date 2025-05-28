import os

import torch

from model.dnn import DNNClassifier
from model.transformer import TransformerClassifier
from utils.data_utils import prepare_eval_loader
from utils.eval_utils import evaluate


def evaluate_model(config):
    print("Running final evaluation on test set...")

    test_loader, vocab = prepare_eval_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = config["model"]["type"]
    if model_type == "dnn":
        model = DNNClassifier(
            vocab_size=len(vocab),
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"]
        ).to(device)
    elif model_type == "transformer":
        
        model = TransformerClassifier(
            vocab_size=len(vocab),
            embedding_dim=config["model"]["embedding_dim"],
            num_classes=config["model"]["num_classes"],
            max_len=config["data"]["max_seq_len"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_path = os.path.join(config["misc"]["save_dir"], config["misc"]["model_file"])
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, f1 = evaluate(model, test_loader, device, model_type=model_type)
    print(f"Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
