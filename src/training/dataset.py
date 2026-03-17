# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from utils import random_crop, center_crop

# class MashupDataset(Dataset):

#     def __init__(self, metadata_csv, root_dir, train=True):

#         self.metadata = pd.read_csv(metadata_csv)
#         self.root_dir = Path(root_dir)
#         self.train = train

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):

#         row = self.metadata.iloc[idx]

#         genre = row["genre"]
#         file = row["file"]
        
#         spec_path = self.root_dir / file
#         # print("Spec Path", spec_path)
#         spec = np.load(spec_path)
        
#         # Crop spectrogram
#         if self.train:
#             spec = random_crop(spec)
#         else:
#             spec = center_crop(spec)

#         # convert to tensor
#         spec = torch.tensor(spec).unsqueeze(0).float()

#         label = torch.tensor(row["label"]).long()
        
#         return spec, label
    
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

        return spec, label