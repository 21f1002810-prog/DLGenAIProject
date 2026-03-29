import torch
import torch.nn as nn
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from model  import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- PARAMETERS ----------------
TARGET_SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
TARGET_LEN = 1024

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


# ---------------- MODEL ----------------
num_classes = 10

model = SimpleCNN(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(
    "/kaggle/input/datasets/sudhanwaabokadee/cnn-model/best_model.pth",
    map_location=DEVICE
))

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

            tensor = torch.tensor(c, dtype=torch.float32)\
            .unsqueeze(0)\
            .to(DEVICE)

            out = model(tensor)

            outputs.append(out)

    outputs = torch.stack(outputs).mean(dim=0)

    pred = torch.argmax(outputs, dim=1).item()

    predictions.append(genre_labels[pred])


# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({
    "id": test_csv["id"],
    "genre": predictions
})

submission.to_csv("cnn_submission.csv", index=False)

print("submission.csv created successfully!")