import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from utils import random_crop, center_crop

class MashupDataset(Dataset):

    def __init__(self, metadata_csv, root_dir, train=True):

        self.metadata = pd.read_csv(metadata_csv)
        self.root_dir = Path(root_dir)
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        genre = row["genre"]
        file = row["file"]
        
        spec_path = self.root_dir / file
        # print("Spec Path", spec_path)
        spec = np.load(spec_path)
        
        # Crop spectrogram
        if self.train:
            spec = random_crop(spec)
        else:
            spec = center_crop(spec)

        # convert to tensor
        spec = torch.tensor(spec).unsqueeze(0).float()

        label = torch.tensor(row["label"]).long()
        
        return spec, label
    
# augmented_data_path=Path(r'D:\Projects\DLGenAi Project\augmented_dataset')
# dataset = MashupDataset(
#     metadata_csv=augmented_data_path / "metadata.csv",
#     root_dir=augmented_data_path,
#     train=True
# )

# spec, label = dataset[200]

# print(spec.shape)
# print(label)
# print(label.dtype)