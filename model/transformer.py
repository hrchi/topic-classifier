import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional = nn.Parameter(torch.randn(1, max_len, embedding_dim))  # learned positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)  # [B, L, D]
        embed = embed + self.positional[:, :embed.size(1), :]  # add positional encoding

        encoded = self.encoder(embed)  # [B, L, D]

        pooled = encoded.mean(dim=1)  # simple mean pooling
        out = self.classifier(pooled)
        return out
