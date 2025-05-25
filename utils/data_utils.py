import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.data_loader import TextDataset, get_data_loaders


def load_ag_news_csv(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["label", "title", "description"]
    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1  # Labels: 1–4 → 0–3
    return df["text"].tolist(), df["label"].tolist()


def prepare_data_loaders(config):
    save_dir = config["misc"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Load and split training data
    texts, labels = load_ag_news_csv(
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    )

    val_ratio = config["data"].get("val_split", 0.1)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_ratio, stratify=labels, random_state=42
    )

    train_loader, val_loader, vocab = get_data_loaders(
        config, train_texts, train_labels, val_texts, val_labels
    )

    model_path = os.path.join(save_dir, config["misc"]["model_file"])
    vocab_path = os.path.join(save_dir, config["misc"]["vocab_file"])

    return train_loader, val_loader, vocab, model_path, vocab_path


def prepare_eval_loader(config):
    # Load vocab
    vocab_path = os.path.join(config["misc"]["save_dir"], config["misc"]["vocab_file"])
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load test set
    texts, labels = load_ag_news_csv(
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    )

    test_ds = TextDataset(texts, labels, vocab, config["data"]["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"])

    return test_loader, vocab
