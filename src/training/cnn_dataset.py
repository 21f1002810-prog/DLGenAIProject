import numpy as np
import torch
from torch.utils.data import Dataset
import os


def spec_augment(
    spec,
    num_time_masks=2,
    num_freq_masks=2,
    max_time_mask_pct=0.2,
    max_freq_mask_pct=0.2,
    replace_with_zero=False
):
    spec = spec.copy()

    freq, time = spec.shape

    fill_value = 0 if replace_with_zero else spec.mean()

    # Time masks
    for _ in range(num_time_masks):
        t_mask = int(np.random.uniform(0.0, max_time_mask_pct) * time)
        if t_mask == 0:
            continue
        t0 = np.random.randint(0, time - t_mask)
        spec[:, t0:t0 + t_mask] = fill_value

    # Frequency masks
    for _ in range(num_freq_masks):
        f_mask = int(np.random.uniform(0.0, max_freq_mask_pct) * freq)
        if f_mask == 0:
            continue
        f0 = np.random.randint(0, freq - f_mask)
        spec[f0:f0 + f_mask, :] = fill_value

    return spec


# ---------------- DATASET ----------------
class ChunkedDataset(Dataset):
    def __init__(self, folder, train=True):
        self.folder = folder
        self.train = train

        self.data_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith("data_")
        ])

        self.label_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith("labels_")
        ])

        # 🔴 Load with mmap (efficient)
        self.data = [np.load(f, mmap_mode='r') for f in self.data_files]
        self.labels = [np.load(f, mmap_mode='r') for f in self.label_files]

        # 🔴 Build index mapping
        self.index_map = []
        for file_idx in range(len(self.data)):
            num_samples = len(self.data[file_idx])
            for sample_idx in range(num_samples):
                self.index_map.append((file_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]

        x = self.data[file_idx][sample_idx]
        y = self.labels[file_idx][sample_idx]

        # 🔴 APPLY AUGMENTATION ONLY DURING TRAINING
        if self.train:
            x = spec_augment(x)

        # 🔴 Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y