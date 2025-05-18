import torch
import torch.nn as nn
import torch.optim as optim
from model.dnn import DNNClassifier
from data.data_loader import get_data_loaders
import pandas as pd
import pickle
import os

# Construct full paths
save_dir = config["misc"]["save_dir"]
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, config["misc"]["model_file"])
vocab_path = os.path.join(save_dir, config["misc"]["vocab_file"])

def train_model(config):
    print("🚀 Starting training...")

    df = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv", header=None)
    df.columns = ["label", "title", "description"]
    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_loader, val_loader, vocab = get_data_loaders(config, texts, labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config["training"]["lr"],
                           weight_decay=config["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["training"]["epochs"]):
        print(f"\n🔁 Epoch {epoch+1}/{config['training']['epochs']}")
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
        print(f"✅ Epoch {epoch+1} done. Avg Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), config["misc"]["save_path"])
    print(f"✅ Model saved to {config['misc']['save_path']}")
    
    with open("vocab.pkl") as f:
      vocab = pickle.dump(vocab, f)
