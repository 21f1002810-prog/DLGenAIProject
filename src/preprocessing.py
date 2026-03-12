import librosa
import numpy as np
import random
TARGET_SR = 44100

def load_audio(file_path, sr=TARGET_SR):
    y, _ = librosa.load(file_path, sr=sr)
    return y

def mix_stems(drums, vocals, bass, others):
    min_len = min(len(drums), len(vocals), len(bass), len(others))

    drums = drums[:min_len]
    vocals = vocals[:min_len]
    bass = bass[:min_len]
    others = others[:min_len]

    # w_drums  = random.uniform(0.5, 1.5)
    # w_vocals = random.uniform(0.5, 1.5)
    # w_bass   = random.uniform(0.5, 1.5)
    # w_other  = random.uniform(0.5, 1.5)

    # mix = (
    #     w_drums * drums +
    #     w_vocals * vocals +
    #     w_bass * bass +
    #     w_other * others
    # )

    mix = (drums + vocals + bass + others)/4

    mix = mix / np.max(np.abs(mix))

    return mix


def mel_spectrogram(audio, sr=44100):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db