import librosa
import numpy as np


def preprocess_wav(data, hparams):
    cut_off_index = -(len(data) % hparams.hop_length)
    if cut_off_index != 0:
        return data[:-(len(data) % hparams.hop_length)]
    else:
        return data


def save_wav(file_name, data, hparams):
    librosa.output.write_wav(file_name, data, hparams.sample_rate)

