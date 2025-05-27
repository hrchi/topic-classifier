from model.dnn import DNNClassifier
from model.transformer import \
    TransformerClassifier  # <- we'll create this next


def build_model(config, vocab_size):
    model_type = config["model"]["type"]

    if model_type == "dnn":
        return DNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"]
        )

    elif model_type == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=config["model"]["embedding_dim"],
            num_classes=config["model"]["num_classes"],
            max_len=config["data"]["max_seq_len"],
            num_heads=config["model"].get("num_heads", 4),
            num_layers=config["model"].get("num_layers", 2),
            dropout=config["model"]["dropout"]
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
