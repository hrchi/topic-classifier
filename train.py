import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from model.dnn import DNNClassifier
from model.transformer import TransformerClassifier
from utils.data_utils import prepare_data_loaders
from utils.early_stopping import EarlyStopping
from utils.eval_utils import evaluate  # ← shared evaluator


def train_model(config):
    print("Starting training...")

    # Load data and paths
    train_loader, val_loader, vocab, model_path, vocab_path = prepare_data_loaders(config)

    # Setup model and training components
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='max',           # because we want to maximize F1
      factor=0.5,           # reduce LR by 50% when plateau hits
      patience=2,           # wait 2 epochs before reducing
    )

    criterion = nn.CrossEntropyLoss()

    best_avg_f1 = 0.0  # For tracking best model
    best_f1 = 0.0  # For tracking best model

    early_stopper = EarlyStopping(patience=config["training"].get("patience", 3), mode="max")

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            if model_type == "transformer":
                x_batch, y_batch, src_key_padding_mask = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)
                logits = model(x_batch, src_key_padding_mask=src_key_padding_mask)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"   Step {i+1}/{len(train_loader)} | Batch loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg training loss: {avg_loss:.4f}")
        #print training scores
        train_acc, train_f1 = evaluate(model, train_loader, device, model_type=model_type)
        print(f"Train Accuracy: {train_acc:.4f}, Train Macro F1: {train_f1:.4f}")

        # Validation check
        model.eval()
        val_acc, val_f1 = evaluate(model, val_loader, device, model_type=model_type)
        print(f"Val Accuracy: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        avg_f1 = (train_f1 + val_f1) / 2
        # Save best model (optional)
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"Best model updated (best average F1 = {best_avg_f1:.4f}), best F1 = {best_f1:.4f}")
        else: #remind us about best
            print(f"Best model updated (best average F1 = {best_avg_f1:.4f}), best F1 = {best_f1:.4f} remains unchanged.")
        #check for early stopping
        if early_stopper(val_f1):
            print("Early stopping triggered.")
            break
    # Save vocab
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Training complete. Best model saved to {model_path}")
