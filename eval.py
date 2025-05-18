import torch
from model.dnn import DNNClassifier
from data.data_loader import get_data_loaders
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
import pickle

from data.data_loader import TextDataset
from torch.utils.data import DataLoader
import re

def evaluate_model(config):
    print("🚀 Starting evaluation...")

    # Construct full paths
    save_dir = config["misc"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, config["misc"]["model_file"])
    vocab_path = os.path.join(save_dir, config["misc"]["vocab_file"])
    with open(vocab_path, "rb") as f:
      vocab = pickle.load(f)

    df = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", header=None)
    df.columns = ["label", "title", "description"]
    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    test_ds = TextDataset(texts, labels, vocab, config["data"]["max_seq_len"])
    val_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"✅ Eval Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # Count <unk> tokens
    unk_token = vocab["<unk>"]
    unk_count = 0
    total_tokens = 0

    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        for tok in tokens:
            total_tokens += 1
            if tok not in vocab:
                unk_count += 1

    unk_ratio = unk_count / total_tokens
    print(f"📉 Unseen tokens: {unk_count} out of {total_tokens} ({unk_ratio:.2%})")

    oov_tokens = [tok for text in texts for tok in re.findall(r"\b\w+\b", text.lower()) if tok not in vocab]
    from collections import Counter
    print(Counter(oov_tokens).most_common(10))

