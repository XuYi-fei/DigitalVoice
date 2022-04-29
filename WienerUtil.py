import librosa
import numpy as np
from config import DefaultConfig


class Wiener:
    def __init__(self):
        self.config = DefaultConfig.weiner_config
        self.file_config = DefaultConfig.file_config
        self.filter = None

    def genFilter(self, noise, data):
        """
        :param noise: a series of noise data or a single noise data (Must have the same length as data)
        :param data: a series of clean data or a single clean data (Must have the same length as noise)
        :return:
        """
        if isinstance(data, list) and isinstance(noise, list):
            Px, Pn = [], []
            for clean_data, noise_data in zip(data, noise):
                S_clean = librosa.stft(clean_data, n_fft=self.config.fft_length,
                                       hop_length=self.config.frame_shift_length, win_length=self.config.window_length)
                S_noise = librosa.stft(noise_data, n_fft=self.config.fft_length,
                                       hop_length=self.config.frame_shift_length, win_length=self.config.window_length)
                Pxx = np.mean((np.abs(S_clean)) ** 2, axis=1, keepdims=True)  # Dx1
                Pnn = np.mean((np.abs(S_noise)) ** 2, axis=1, keepdims=True)
                Px.append(Pxx)
                Pn.append(Pnn)
            train_Pxx = np.mean(np.concatenate(Px, axis=1), axis=1, keepdims=True)
            train_Pnn = np.mean(np.concatenate(Pn, axis=1), axis=1, keepdims=True)

            self.filter = (train_Pxx / (train_Pxx + self.config.alpha * train_Pnn)) ** self.config.beta
        elif isinstance(data, np.ndarray) and isinstance(noise, np.ndarray):
            S_clean = librosa.stft(data, n_fft=self.config.fft_length,
                                   hop_length=self.config.frame_shift_length, win_length=self.config.window_length)
            S_noise = librosa.stft(noise, n_fft=self.config.fft_length,
                                   hop_length=self.config.frame_shift_length, win_length=self.config.window_length)
            Pxx = np.mean((np.abs(S_clean)) ** 2, axis=1, keepdims=True)
            Pnn = np.mean((np.abs(S_noise)) ** 2, axis=1, keepdims=True)
            self.filter = (Pxx / (Pxx + self.config.alpha * Pnn)) ** self.config.beta

    def waveFiltering(self, data, file_name="filtered.wav"):
        """
        :param file_name: file name for saved filtered data
        :param data: data with noise
        :return: data filtered by wiener filter
        """
        file_name = self.file_config.file_output_path + file_name
        S_data = librosa.stft(data, n_fft=self.config.fft_length,
                              hop_length=self.config.frame_shift_length, win_length=self.config.window_length)
        S_filtered = S_data * self.filter
        filtered_data = librosa.istft(S_filtered, hop_length=self.config.frame_shift_length,
                                      win_length=self.config.window_length)
        return filtered_data
