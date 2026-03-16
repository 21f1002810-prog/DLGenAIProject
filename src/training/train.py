import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import MashupDataset
from model import GenreCNN
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()
    print("Model training started")
    total_loss = 0
    # print("loaderr:",loader)
    for specs, labels in loader:
        

        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(specs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        # print("Eond of train one epoch")
    return total_loss / len(loader)


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

    f1 = f1_score(all_labels, all_preds, average="macro")

    return f1

def main():
    augmented_data_path=Path(r'D:\Projects\DLGenAi Project\augmented_dataset')
    train_dataset = MashupDataset(
        metadata_csv="train_metadata.csv",
        root_dir=augmented_data_path,
        train=True
    )

    val_dataset = MashupDataset(
        metadata_csv="val_metadata.csv",
        root_dir=augmented_data_path,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    print("Dataset segreagated:")

    model = GenreCNN(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    best_f1 = 0

    for epoch in range(epochs):
        print("Epochs started")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )
        print("After train loss")
        val_f1 = validate(
            model,
            val_loader
        )
        print("After validate")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("Model checkpoint saved!")

if __name__ == "__main__":
    main()