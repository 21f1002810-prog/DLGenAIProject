import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import librosa

from transformers import AutoFeatureExtractor
from mashup_generator import generate_mashup

GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

class ASTDataset(Dataset):

    def __init__(self, stems_root, samples_per_epoch=10000):

        self.stems_root = Path(stems_root)
        self.samples_per_epoch = samples_per_epoch

        # Load once → avoids repeated downloads
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        genre = random.choice(GENRES)
        genre_path = self.stems_root / genre

        audio = generate_mashup(genre_path)

        # CRITICAL: resample to 16kHz
        audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].squeeze(0)

        label = GENRES.index(genre)

        return input_values, torch.tensor(label).long()