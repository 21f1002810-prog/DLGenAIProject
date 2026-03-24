import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re


def spec_augment(spec, time_mask=40, freq_mask=20):

    spec = spec.copy()

    F, T = spec.shape

    # time mask
    t = np.random.randint(0, time_mask)
    if T - t > 0:
        t0 = np.random.randint(0, T - t)
        spec[:, t0:t0+t] = spec.mean()

    # freq mask
    f = np.random.randint(0, freq_mask)
    if F - f > 0:
        f0 = np.random.randint(0, F - f)
        spec[f0:f0+f, :] = spec.mean()

    return spec


class ASTDataset(Dataset):

    def __init__(self, folder, train=True):

        self.train = train

        data_files = {}
        label_files = {}

        for f in os.listdir(folder):

            if f.startswith("data_"):
                idx = int(re.findall(r"\d+", f)[0])
                data_files[idx] = os.path.join(folder, f)

            if f.startswith("labels_"):
                idx = int(re.findall(r"\d+", f)[0])
                label_files[idx] = os.path.join(folder, f)

        common_idx = sorted(set(data_files) & set(label_files))

        self.data = []
        self.labels = []

        for i in common_idx:

            d = np.load(data_files[i], mmap_mode="r")
            l = np.load(label_files[i], mmap_mode="r")

            self.data.append(d)
            self.labels.append(l)

        self.index_map = []

        for file_idx in range(len(self.data)):

            n = len(self.data[file_idx])

            for sample_idx in range(n):

                self.index_map.append((file_idx, sample_idx))

        print("Total samples:", len(self.index_map))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        file_idx, sample_idx = self.index_map[idx]

        spec = self.data[file_idx][sample_idx].astype(np.float32)
        label = int(self.labels[file_idx][sample_idx])

        # normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if self.train:
            spec = spec_augment(spec)

        spec = torch.tensor(spec)

        return spec, torch.tensor(label).long()