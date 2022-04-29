import numpy as np
from scipy.signal import lfilter, firwin
from config import DefaultConfig


class Noise:
    def __init__(self):
        self.noise = None
        self.config = DefaultConfig.noise_config

    def addGaussianNoise(self, data):
        """
        :param data: 输入的音频序列数据
        :param noise: 给定的噪声信号
        :return:
        """
        if not len(data) == len(self.noise):
            print("")
            return False

        # 原数据的能量
        p_clean = np.sum(np.abs(data) ** 2)
        # 噪声信号的能量
        p_noise = np.sum(np.abs(self.noise) ** 2)
        # 计算缩放因子
        scale = np.sqrt((p_clean / p_noise) * np.power(10, -self.config.snr / 10))
        return data + scale * self.noise

    def generateGaussianNoise(self, N, order_filter, fs):
        noise = np.random.randn(N)
        FIR_filter = firwin(order_filter, [2 * self.config.f_l / fs, 2 * self.config.f_h / fs], pass_zero="bandpass")
        noise = lfilter(FIR_filter, 1.0, noise)
        self.noise = noise
