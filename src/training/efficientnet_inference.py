import torch
import torch.nn as nn
import librosa
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ---------------- PARAMETERS ----------------

TARGET_SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
TARGET_LEN = 1024
NUM_CLASSES = 10


# ---------------- FEATURE EXTRACTION ----------------

def wav_to_spectrogram(file_path):

    audio, sr = librosa.load(file_path, sr=TARGET_SR)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel)

    mel_db = mel_db.astype(np.float32)

    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)

    return mel_db


# ---------------- MULTI CROP ----------------

def get_crops(spec):

    T = spec.shape[1]

    if T <= TARGET_LEN:

        pad = TARGET_LEN - T

        spec = np.pad(spec, ((0,0),(0,pad)))

        return [spec]

    crops = []

    # left
    crops.append(spec[:, :TARGET_LEN])

    # center
    start = (T - TARGET_LEN)//2
    crops.append(spec[:, start:start+TARGET_LEN])

    # right
    crops.append(spec[:, -TARGET_LEN:])

    return crops


# ---------------- PREPROCESS FOR EFFICIENTNET ----------------

def preprocess_spec(spec):

    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

    spec_min = spec.min()
    spec_max = spec.max()

    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)

    spec = cv2.resize(spec, (224,224), interpolation=cv2.INTER_LINEAR)

    spec = np.stack([spec, spec, spec], axis=0)

    tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

    return tensor


# ---------------- MODEL ----------------

class EfficientNetClassifier(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=False)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        return self.backbone(x)


model = EfficientNetClassifier(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        "/kaggle/working/efficientnet_best.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ---------------- LABEL MAP ----------------

genre_labels = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


# ---------------- LOAD TEST ----------------

test_csv = pd.read_csv(
    "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv"
)

mashup_folder = Path(
    "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/mashups"
)


# ---------------- INFERENCE ----------------

predictions = []

for _, row in test_csv.iterrows():

    file_path = mashup_folder / Path(row["filename"]).name

    spec = wav_to_spectrogram(file_path)

    crops = get_crops(spec)

    outputs = []

    with torch.no_grad():

        for c in crops:

            tensor = preprocess_spec(c).to(DEVICE)

            logits = model(tensor)

            outputs.append(logits)

    outputs = torch.stack(outputs).mean(dim=0)

    pred = torch.argmax(outputs, dim=1).item()

    predictions.append(genre_labels[pred])


# ---------------- SUBMISSION ----------------

submission = pd.DataFrame({
    "id": test_csv["id"],
    "genre": predictions
})

submission.to_csv("efficientnet_submission.csv", index=False)

print("submission.csv created successfully!")