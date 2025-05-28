import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, data_loader, device,model_type="dnn"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
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
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1