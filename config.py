from configs import *


class FileConfig:
    def __init__(self):
        self.file_path = FILE_PATH
        self.file_output_path = FILE_OUTPUT_PATH


class NoiseConfig:
    def __init__(self, order_filter=128, kind=1):
        self.snr = NOISE_SNR
        self.kind = 1
        self.f_l = NOISE_FREQUENCY_LOW
        self.f_h = NOISE_FREQUENCY_HIGH
        self.order_filter = order_filter


class WienerConfig:
    def __init__(self):
        self.fft_length = WIENER_N_FFT
        self.alpha = WIENER_ALPHA
        self.beta = WIENER_BETA
        self.frame_shift_length = WIENER_FRAME_SHIFT_LENGTH
        self.window_length = WIENER_WINDOW_LENGTH


class SubFilterConfig:
    def __init__(self):
        self.fft_length = SUB_FILTER_N_FFT
        self.alpha = SUB_FILTER_ALPHA
        self.beta = SUB_FILTER_BETA
        self.gamma = SUB_FILTER_GAMMA
        self.frame_shift_length = SUB_FILTER_FRAME_SHIFT_LENGTH
        self.window_length = SUB_FILTER_WINDOW_LENGTH
        self.k = SUB_FILTER_K


class AudioConfig:
    def __init__(self):
        self.clean_data_path = AUDIO_CLEAN_DATA_FILE_PATH
        self.noise_data_path = AUDIO_NOISE_DATA_FILE_PATH
        self.noisy_data_path = AUDIO_NOISY_DATA_FILE_PATH


class Config:
    def __init__(self):
        self.noise_config = NoiseConfig()
        self.weiner_config = WienerConfig()
        self.audio_config = AudioConfig()
        self.file_config = FileConfig()
        self.sub_filter_config = SubFilterConfig()


DefaultConfig = Config()
if not os.path.isdir(DefaultConfig.file_config.file_path):
    os.makedirs(FILE_PATH)
if not os.path.isdir(DefaultConfig.file_config.file_output_path):
    os.makedirs(DefaultConfig.file_config.file_output_path)
