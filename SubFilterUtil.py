import librosa
from config import DefaultConfig
import numpy as np


class SubFilterUtil:
    def __init__(self):
        self.config = DefaultConfig.sub_filter_config

    def sub_proc(self, noise_data):
        """
        :param noise_data: data mixed with noise
        :param noise: pure noise
        :return:
        """
        noise = noise_data[:self.config.window_length*30]
        # 计算混合了噪声的数据的相关参数
        S_noise_data = librosa.stft(noise_data, n_fft=self.config.fft_length, hop_length=self.config.frame_shift_length,
                                    win_length=self.config.window_length)
        D, T = np.shape(S_noise_data)
        Magnitude_noise_data = np.abs(S_noise_data)
        Phase_noise_data = np.angle(S_noise_data)
        Power_noise_data = Magnitude_noise_data ** 2

        # 计算纯噪声的数据的相关参数（通常取未知待处理数据的前一小段部分作为噪声）
        S_noise = librosa.stft(noise, n_fft=self.config.fft_length, hop_length=self.config.frame_shift_length,
                               win_length=self.config.window_length)
        Magnitude_noise = np.mean(np.abs(S_noise), axis=1, keepdims=True)
        Power_noise = Magnitude_noise ** 2
        Power_noise = np.tile(Power_noise, [1, T])

        # 引入部分平滑
        Magnitude_noise_data_new = np.copy(Magnitude_noise_data)
        for t in range(1, T - self.config.k):
            Magnitude_noise_data_new[:, t] = np.mean(Magnitude_noise_data[:, t - self.config.k:t + 1 + self.config.k],
                                                     axis=1)
        Power_noise_data = Magnitude_noise_data_new ** 2

        # 去噪
        Power_filtered_data = np.power(Power_noise_data, self.config.gamma) \
                              - self.config.alpha * np.power(Power_noise, self.config.gamma)
        Power_filtered_data = np.power(Power_filtered_data, 1 / self.config.gamma)

        # 找到值过小的部分
        mask = (Power_filtered_data >= self.config.beta * Power_noise) - 0
        Power_filtered_data = mask * Power_filtered_data + self.config.beta * (1 - mask) * Power_noise
        Magnitude_filtered_data = np.sqrt(Power_filtered_data)
        Magnitude_filtered_data_new = np.copy(Magnitude_filtered_data)

        # 计算最大残差
        max_nr = np.max(np.abs(S_noise_data[:, :31]) - Magnitude_noise, axis=1)

        for t in range(1, T - 1):
            index = np.where(Magnitude_filtered_data[:, t] < max_nr)[0]
            temp = np.min(Magnitude_filtered_data[:, t - 1:t + 2], axis=1)
            Magnitude_filtered_data_new[index, t] = temp[index]

        S_filtered_data = Magnitude_filtered_data_new * np.exp(1j * Phase_noise_data)
        filtered_data = librosa.istft(S_filtered_data, hop_length=self.config.frame_shift_length,
                                      win_length=self.config.window_length)
        return filtered_data
