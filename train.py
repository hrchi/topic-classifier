import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from model.dnn import DNNClassifier
from utils.data_utils import prepare_data_loaders
from utils.early_stopping import EarlyStopping
from utils.eval_utils import evaluate  # ← shared evaluator


def train_model(config):
    print("Starting training...")

    # Load data and paths
    train_loader, val_loader, vocab, model_path, vocab_path = prepare_data_loaders(config)

    # Setup model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0  # For tracking best model
    early_stopper = EarlyStopping(patience=config["training"].get("patience", 3), mode="max")

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        model.train()
        total_loss = 0

        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"   Step {i+1}/{len(train_loader)} | Batch loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg training loss: {avg_loss:.4f}")

        # 🔍 Validation check
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Val Accuracy: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}")

        # Save best model (optional)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"Best model updated (F1 = {best_f1:.4f})")
        #check for early stopping
        if early_stopper(val_f1):
            print("Early stopping triggered.")
            break
    # Save vocab
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Training complete. Best model saved to {model_path}")