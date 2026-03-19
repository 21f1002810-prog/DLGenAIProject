import librosa
import numpy as np
import random
from pathlib import Path

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

    w_drums  = random.uniform(0.6, 1.5)
    w_vocals = random.uniform(0.6, 1.5)
    w_bass   = random.uniform(0.6, 1.5)
    w_other  = random.uniform(0.6, 1.5)

    mix = (
        w_drums * drums +
        w_vocals * vocals +
        w_bass * bass +
        w_other * others
    )

    mix = mix / np.max(np.abs(mix))

    return mix


def random_time_stretch(audio):
    rate = random.uniform(0.8, 1.25)
    stretched = librosa.effects.time_stretch(audio,rate=rate)
    return stretched

def add_noise(audio, noise_files):
    noise_file = random.choice(noise_files)
    noise, _ = librosa.load(noise_file, sr=TARGET_SR)

    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)

    noise = noise[:len(audio)]
    noise_level = random.uniform(0.01, 0.05)
    noisy_audio = audio + noise_level * noise
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))

    return noisy_audio

def mel_spectrogram(audio, sr=TARGET_SR):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db