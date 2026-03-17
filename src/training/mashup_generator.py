import random
from pathlib import Path
import librosa
import numpy as np
import os,sys
from preprocessing  import load_audio, mix_stems

def generate_mashup(genre_path):
    print("Genre Mashup:", genre_path)
    songs = list(genre_path.iterdir())

    s1, s2, s3, s4 = random.sample(songs, 4)
    # print(s1,s2,s3,s4)
    drums = load_audio(s1 / "drums.wav")
    vocals = load_audio(s2 / "vocals.wav")
    bass = load_audio(s3 / "bass.wav")
    others = load_audio(s4 / "other.wav")

    mashup = mix_stems(drums, vocals, bass, others)

    return mashup

def random_time_stretch(audio):
    rate = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched

def apply_random_gain(stem):
    gain = random.uniform(0.5, 1.5)
    return stem * gain

def add_noise(audio, noise_files):

    noise_file = random.choice(noise_files)

    noise, _ = librosa.load(noise_file, sr=44100)

    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)

    noise = noise[:len(audio)]

    snr = random.uniform(5, 20)
    audio_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(audio_power / (10**(snr/10) * noise_power))
    noisy = audio + scale * noise
    return noisy


