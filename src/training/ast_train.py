import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import ASTForAudioClassification
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from ast_dataset import ASTDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ---------------- TRAIN ----------------
def train_epoch(model, loader, optimizer, scaler):

    model.train()

    total_loss = 0

    loop = tqdm(loader)

    for x, y in loop:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast():

            outputs = model(x)

            loss = nn.CrossEntropyLoss()(outputs.logits, y)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# ---------------- VALIDATE ----------------
def validate(model, loader):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for x, y in tqdm(loader):

            x = x.to(DEVICE)

            outputs = model(x)

            p = torch.argmax(outputs.logits, dim=1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    return f1, acc


# ---------------- MAIN ----------------
def main():

    wandb.init(
    project="21f1002810-t12026",
    name="ast_model_run",
    config={
        "model": "AST",
        "epochs": 12,
        "batch_size": 12,
        "optimizer": "AdamW",
        "lr": 2e-5,
        "scheduler": "CosineAnnealingLR"
    }
    )
    # wandb.watch(model, log=None)
    

    dataset_path = "/kaggle/input/datasets/sudhanwaabokadee/audio-genre-processed"

    full_dataset = ASTDataset(dataset_path, train=True)

    # split
    idx = np.arange(len(full_dataset))
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.8 * len(idx))

    train_idx = idx[:split]
    val_idx = idx[split:]

    train_dataset = Subset(full_dataset, train_idx)

    val_dataset = Subset(
        ASTDataset(dataset_path, train=False),
        val_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=10,
    ignore_mismatched_sizes=True
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    scaler = GradScaler()

    best_f1 = 0

    epochs = 12

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        val_f1, val_acc = validate(model, val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Macro F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
            model.state_dict(),
            "ast_best.pth"
            )

            wandb.save("ast_best.pth")

            print("Model saved")

    print("Best F1:", best_f1)
    wandb.finish()


if __name__ == "__main__":
    main()