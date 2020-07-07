class AudioParams():
    def __init__(self,
                 sample_rate=22050,
                 hop_length=256,
                 n_fft=1024,
                 n_mels=80,
                 window='hann',
                 win_length=1024
                 ):

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.window = window
        self.win_length = win_length
