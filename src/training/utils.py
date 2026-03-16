import numpy as np

def center_crop(spec, target_len=1024):

        time_len = spec.shape[1]

        if time_len > target_len:
            start = (time_len - target_len) // 2
            spec = spec[:, start:start + target_len]

        else:
            pad = target_len - time_len
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")

        return spec

def random_crop(spec, target_len=1024):

    time_len = spec.shape[1]

    if time_len > target_len:

        start = np.random.randint(0, time_len - target_len)
        spec = spec[:, start:start+target_len]

    else:

        pad = target_len - time_len
        spec = np.pad(spec, ((0,0),(0,pad)), mode="constant")

    return spec