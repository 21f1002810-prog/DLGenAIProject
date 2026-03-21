import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from dataset import ChunkedDataset
from model import SimpleCNN
from pathlib import Path
from torch.amp import autocast, GradScaler
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE=str(DEVICE)
print("Using device:", DEVICE)

# ---------------- TRAIN ----------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for specs, labels in loader:
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type=DEVICE):
            outputs = model(specs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------- VALIDATE ----------------
def validate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(specs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")


# ---------------- MAIN ----------------
def main():

    dataset_path = Path("/kaggle/working/data/audio-genre-processed")
    dataset = ChunkedDataset(dataset_path)

    # 🔴 FIXED split (deterministic)
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 🔴 FIX workers (Kaggle safe)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 🔴 Model
    model = SimpleCNN(num_classes=10).to(DEVICE)

    # 🔴 Optimizer (with weight decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )

    # 🔴 Scheduler (CORRECT placement)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=DEVICE)

    epochs = 20
    best_f1 = 0
    patience = 5
    counter = 0

    # 🔴 Sanity check
    x, y = next(iter(train_loader))
    print("Sample batch shape:", x.shape)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        val_f1 = validate(model, val_loader)

        # 🔴 CORRECT scheduler usage
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        # 🔴 Early stopping + checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0

            torch.save(model.state_dict(), "best_model.pth")
            print("Model checkpoint saved!")

        else:
            counter += 1
            print(f"No improvement. Early stop counter: {counter}/{patience}")

            if counter >= patience:
                print("Early stopping triggered!")
                break

    print("Training complete!")


if __name__ == "__main__":
    main()