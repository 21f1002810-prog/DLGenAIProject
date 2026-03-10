import torch
from torch.utils.data import Dataset
import random
from mashup_generator import generate_mashup,add_noise,random_time_stretch,apply_random_gain
from feature_extraction import mel_spectrogram

class MashupDataset(Dataset):

    def __init__(self, genre_paths, noise_files):
        self.genre_paths = genre_paths
        self.noise_files = noise_files

        GENRES = [
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

        GENRE_TO_ID = {genre: i for i, genre in enumerate(GENRES)}
        ID_TO_GENRE = {i: genre for genre, i in GENRE_TO_ID.items()}

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        genre = random.choice(list(self.genre_paths.keys()))
        genre_path = self.genre_paths[genre]

        audio = generate_mashup(genre_path)

        audio = random_time_stretch(audio)

        audio = add_noise(audio, self.noise_files)

        mel = mel_spectrogram(audio)

        mel = torch.tensor(mel).unsqueeze(0)

        label = self.GENRE_TO_ID[genre]

        return mel.float(), label