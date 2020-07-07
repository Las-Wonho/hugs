import librosa
import numpy as np


def wav_to_mels(data, hparams):
    return librosa.feature.melspectrogram(data[:-1],
                                          sr=hparams.sample_rate,
                                          n_fft=hparams.n_fft,
                                          hop_length=hparams.hop_length,
                                          win_length=hparams.win_length,
                                          window=hparams.window,
                                          n_mels=hparams.n_mels).astype(np.float32)


def save_mels(file_name, mels):
    np.save(file_name, mels.T)