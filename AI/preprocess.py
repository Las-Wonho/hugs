import os

import multiprocessing

from hparams.AudioParams import AudioParams
from preprocess.audio import *
from preprocess.mels import *
from utils.audio import *


def preprocess(file_name, hparams):
    wav_data = load_wav('./data/music/' + file_name, hparams)

    preprocessed_wav = preprocess_wav(wav_data, hparams)
    mels = wav_to_mels(preprocessed_wav, hparams)

    save_wav('./preprocessed_data/wav/' + file_name, preprocessed_wav, hparams)
    save_mels('./preprocessed_data/mels/' + file_name[:-4], mels)
    print(file_name)
    raise ValueError


if __name__ == '__main__':
    audio_params = AudioParams()

    wav_list = os.listdir('./data/music')
    n_list = 100
    list_split_position = [0]

    for i in range(n_list - 1):
        list_split_position.append((len(wav_list) // n_list) * (i + 1))

    list_split_position.append(-1)

    for i in range(n_list):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.starmap(preprocess, [(file_name, audio_params) for file_name in
                                  wav_list[list_split_position[i]:list_split_position[i + 1]]])
