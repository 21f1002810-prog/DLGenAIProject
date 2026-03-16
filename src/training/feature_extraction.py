import librosa
import numpy as np

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