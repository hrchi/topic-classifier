import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens][:self.max_len]
        return ids + [self.vocab["<pad>"]] * (self.max_len - len(ids))

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

    train_ds = TextDataset(train_texts, train_labels, vocab, config["data"]["max_seq_len"])
    val_ds = TextDataset(val_texts, val_labels, vocab, config["data"]["max_seq_len"])

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"])

    return train_loader, val_loader, vocab
