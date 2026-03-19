import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path

from mashup_generator import generate_mashup
from preprocessing import random_time_stretch, add_noise, mel_spectrogram
from feature_extraction import mel_spectrogram
from utils import random_crop

GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

class MashupDataset(Dataset):

    def __init__(self, stems_root, noise_root, samples_per_epoch=20000):

        self.stems_root = Path(stems_root)
        self.noise_files = list(Path(noise_root).glob("**/*.wav"))
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        # pick random genre
        genre = random.choice(GENRES)
        genre_path = self.stems_root / genre

        # generate mashup
        audio = generate_mashup(genre_path)

        # tempo augmentation
        audio = random_time_stretch(audio)

        # noise augmentation
        if random.random() < 0.7:
            audio = add_noise(audio, self.noise_files)

        # spectrogram
        spec = mel_spectrogram(audio)

        # crop
        spec = random_crop(spec)

        spec = torch.tensor(spec).unsqueeze(0).float()

        label = GENRES.index(genre)
        label = torch.tensor(label).long()
        # print(genre_path)
        # print(spec,label)
        # print("\n")
        return spec, label

# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import random
# from pathlib import Path

# from preprocessing import (
#     load_audio,
#     mix_stems,
#     random_time_stretch,
#     add_noise,
#     mel_spectrogram
# )

# from utils import random_crop

# GENRES = [
#     "blues","classical","country","disco","hiphop",
#     "jazz","metal","pop","reggae","rock"
# ]


# class MashupDataset(Dataset):

#     def __init__(self, stems_root, noise_root, samples_per_epoch=20000):

#         self.stems_root = Path(stems_root)
#         self.samples_per_epoch = samples_per_epoch

#         # preload noise
#         self.noise_files = list(Path(noise_root).glob("**/*.wav"))

#         print("Loading stems into RAM...")

#         self.stem_cache = self._load_stems()

#         print("Finished loading stems.")

#     def _load_stems(self):

#         stem_cache = {}

#         for genre in GENRES:

#             genre_path = self.stems_root / genre

#             songs = list(genre_path.iterdir())

#             genre_stems = []

#             for song in songs:

#                 try:
#                     stems = {
#                         "drums": load_audio(song / "drums.wav"),
#                         "vocals": load_audio(song / "vocals.wav"),
#                         "bass": load_audio(song / "bass.wav"),
#                         "other": load_audio(song / "other.wav"),
#                     }

#                     genre_stems.append(stems)

#                 except Exception:
#                     continue

#             stem_cache[genre] = genre_stems

#         return stem_cache

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):

#         # pick random genre
#         genre = random.choice(GENRES)

#         songs = self.stem_cache[genre]

#         # pick 4 random songs (one per stem)
#         drums  = random.choice(songs)["drums"]
#         vocals = random.choice(songs)["vocals"]
#         bass   = random.choice(songs)["bass"]
#         other  = random.choice(songs)["other"]

#         # mix
#         audio = mix_stems(drums, vocals, bass, other)

#         # augment
#         audio = random_time_stretch(audio)

#         if random.random() < 0.7:
#             audio = add_noise(audio, self.noise_files)

#         # spectrogram
#         spec = mel_spectrogram(audio)

#         # crop
#         spec = random_crop(spec)

#         spec = torch.tensor(spec).unsqueeze(0).float()

#         label = GENRES.index(genre)
#         label = torch.tensor(label).long()

#         return spec, label