import numpy as np
import os
from tqdm import tqdm

def fix_length(spec, target_len=1024):
    time_len = spec.shape[1]

    if time_len > target_len:
        start = np.random.randint(0, time_len - target_len)
        spec = spec[:, start:start+target_len]
    else:
        pad = target_len - target_len if time_len > target_len else target_len - time_len
        spec = np.pad(spec, ((0,0),(0,pad)), mode="constant")

    return spec

root = r"augmented_dataset"
genres = sorted([g for g in os.listdir(root) if os.path.isdir(os.path.join(root, g))])

label_map = {genre: i for i, genre in enumerate(genres)}

chunk_size = 2000
data_chunk = []
label_chunk = []
chunk_id = 0

os.makedirs("processed", exist_ok=True)

# 🔴 STEP 1: collect all file paths first (IMPORTANT)
all_files = []
for genre in genres:
    genre_path = os.path.join(root, genre)
    for file in os.listdir(genre_path):
        if file.endswith(".npy"):
            all_files.append((genre, os.path.join(genre_path, file)))

print(f"Total files: {len(all_files)}")

# 🔴 STEP 2: progress bar over entire dataset
for genre, path in tqdm(all_files, desc="Processing", unit="files"):
    
    arr = np.load(path)
    arr = fix_length(arr, target_len=1024)

    data_chunk.append(arr)
    label_chunk.append(label_map[genre])

    if len(data_chunk) == chunk_size:
        np.save(f"processed/data_{chunk_id}.npy", np.stack(data_chunk))
        np.save(f"processed/labels_{chunk_id}.npy", np.array(label_chunk))

        data_chunk = []
        label_chunk = []
        chunk_id += 1

# save remaining
if data_chunk:
    np.save(f"processed/data_{chunk_id}.npy", np.stack(data_chunk))
    np.save(f"processed/labels_{chunk_id}.npy", np.array(label_chunk))

print("Done")