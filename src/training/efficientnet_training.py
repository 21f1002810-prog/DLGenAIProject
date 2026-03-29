import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score

from efficientnet_dataset import SpectrogramDataset
from efficientnet_model import EfficientNetClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
wandb.init(
    project="21f1002810-t12026",
    name="efficientnet_run",
    config={
        "model": "EfficientNet",
        "epochs": 20,
        "batch_size": 64,
        "optimizer": "AdamW",
        "lr": 3e-4
    }
)


DATA_FOLDER = "/kaggle/input/datasets/sudhanwaabokadee/audio-genre-processed"

dataset = SpectrogramDataset(DATA_FOLDER, train=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
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
    pin_memory=True,
    persistent_workers=True
)

model = EfficientNetClassifier(num_classes=10).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


best_acc = 0


for epoch in range(20):

    model.train()

    running_loss = 0

    for x, y in tqdm(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Train Loss:", running_loss/len(train_loader))


    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("Validation Accuracy:", acc)
    wandb.log({
    "epoch": epoch + 1,
    "train_loss": running_loss / len(train_loader),
    "val_accuracy": acc,
    "val_f1": f1,
    "learning_rate": optimizer.param_groups[0]["lr"]
    })

    if acc > best_acc:

        best_acc = acc

        torch.save(model.state_dict(), "efficientnet_best.pth")
        wandb.save("efficientnet_best.pth")

        print("Model saved!")

wandb.finish()
print("Training complete")