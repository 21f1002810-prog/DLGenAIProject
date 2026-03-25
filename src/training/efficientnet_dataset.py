import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import cv2


class SpectrogramDataset(Dataset):

    def __init__(self, folder, train=True):

        self.data = []
        self.labels = []

        for i in range(10):

            x = np.load(os.path.join(folder, f"data_{i}.npy"))
            y = np.load(os.path.join(folder, f"labels_{i}.npy"))

            self.data.append(x)
            self.labels.append(y)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        self.train = train

    def __len__(self):
        return len(self.data)

    def random_crop(self, spec):

        target_len = 1024
        T = spec.shape[1]

        if T <= target_len:
            pad = target_len - T
            spec = np.pad(spec, ((0,0),(0,pad)))
            return spec

        start = random.randint(0, T - target_len)

        return spec[:, start:start+target_len]

    def __getitem__(self, idx):

        spec = self.data[idx]
        label = self.labels[idx]

        if self.train:
            spec = self.random_crop(spec)

        # convert to float32
        spec = spec.astype(np.float32)

        # remove bad values
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        # min-max normalize (stable)
        spec_min = spec.min()
        spec_max = spec.max()

        if spec_max - spec_min > 0:
            spec = (spec - spec_min) / (spec_max - spec_min)

        # resize
        spec = cv2.resize(spec, (224, 224), interpolation=cv2.INTER_LINEAR)

        # convert to 3 channels
        spec = np.stack([spec, spec, spec], axis=0)

        spec = torch.tensor(spec, dtype=torch.float32)

        label = torch.tensor(label).long()

        return spec, label