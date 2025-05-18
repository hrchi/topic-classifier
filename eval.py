import torch
from model.dnn import DNNClassifier
from data.data_loader import get_data_loaders
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def evaluate_model(config):
    print("🚀 Starting evaluation...")

    df = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", header=None)
    df.columns = ["label", "title", "description"]
    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    _, val_loader, vocab = get_data_loaders(config, texts, labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(config["misc"]["save_path"]))
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
