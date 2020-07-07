import librosa


def load_wav(file_name, hparams):
    return librosa.load(file_name, hparams.sample_rate)[0]
