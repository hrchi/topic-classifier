import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional = nn.Parameter(torch.randn(1, max_len, embedding_dim)*0.01)  # learned positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embedding_dim,
            nhead = num_heads,
            dropout = dropout,
            batch_first = True #[B, L, D] format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, src_key_padding_mask=None) -> torch.Tensor:
        # x: [batch_size, seq_len]
        embed = self.embedding(x)  # [B, L, D]
        embed = embed + self.positional[:, :embed.size(1), :]  # add positional encoding

        # Apply attention mask
        encoded = self.encoder(embed, src_key_padding_mask=src_key_padding_mask)  # [B, L, D]

                # Masked mean pooling
        if src_key_padding_mask is not None:
            # Mask: False = keep, True = pad → invert for pooling
            mask = ~src_key_padding_mask  # [B, L], now True = valid token
            mask = mask.unsqueeze(-1).type_as(encoded)  # [B, L, 1]
            summed = (encoded * mask).sum(dim=1)        # [B, D]
            count = mask.sum(dim=1).clamp(min=1e-9)     # avoid division by zero
            pooled = summed / count                     # [B, D]
        else:
            pooled = encoded.mean(dim=1)

        out = self.classifier(pooled)
        return out
if __name__ == "__main__":
    import torch

    vocab_size = 1000
    embedding_dim = 128
    num_classes = 4
    max_len = 100
    batch_size = 4
    seq_len = 30  # Simulate a batch with shorter sequences

    # Instantiate model
    model = TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        max_len=max_len,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )

    # Dummy input (simulate tokens, with padding id = 0)
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))  # skip 0 and 1 for <pad> and <unk>
    input_ids[:, -2:] = 0  # simulate some padding at the end

    # Padding mask: True where padding
    src_key_padding_mask = (input_ids == 0)

    # Forward pass
    output = model(input_ids, src_key_padding_mask=src_key_padding_mask)

    print("Input shape:", input_ids.shape)
    print("Mask shape:", src_key_padding_mask.shape)
    print("Output shape:", output.shape)  # Should be [batch_size, num_classes]
