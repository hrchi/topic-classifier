import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from data.collate import transformer_collate_fn


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len, pad_inputs=True):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_inputs = pad_inputs 

    def encode(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens][:self.max_len]


        if self.pad_inputs:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))

        return ids[:self.max_len]  # always truncate just in case

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.texts)

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def get_data_loaders(config, train_texts, train_labels, val_texts, val_labels):
    vocab = build_vocab(train_texts)
    pad_idx = vocab["<pad>"]
    model_type = config["model"]["type"]

    padding_mode = config["data"].get("padding", "auto")

    if padding_mode == "auto":
        pad_inputs = model_type == "dnn"
    else:
        pad_inputs = (padding_mode == "static")

    train_ds = TextDataset(train_texts, train_labels, vocab, config["data"]["max_seq_len"], pad_inputs)
    val_ds = TextDataset(val_texts, val_labels, vocab, config["data"]["max_seq_len"], pad_inputs)

    if model_type == "transformer":
        collate = transformer_collate_fn(pad_idx)
        train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"])

    return train_loader, val_loader, vocab
