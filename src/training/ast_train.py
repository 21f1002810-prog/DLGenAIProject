import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForAudioClassification

from ast_dataset import ASTDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, scaler):

    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Training")

    for inputs, labels in loop:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in tqdm(loader, desc="Validation"):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")


def main():

    project_root = Path(__file__).resolve().parents[2]

    stems_root = project_root / "dataset" / "genres_stems"

    train_dataset = ASTDataset(stems_root, samples_per_epoch=8000)
    val_dataset   = ASTDataset(stems_root, samples_per_epoch=2000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,   # SAFE
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10
    ).to(DEVICE)

    # Phase 1: Freeze backbone
    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0

    print("\n--- Phase 1: Training classifier ---")

    for epoch in range(5):

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_f1 = validate(model, val_loader)

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "ast_best.pth")

    # Phase 2: Unfreeze everything
    print("\n--- Phase 2: Full fine-tuning ---")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_f1 = validate(model, val_loader)

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "ast_best.pth")


if __name__ == "__main__":
    main()