import librosa
import soundfile as sf
from config import DefaultConfig


class AudioUtils:
    def __init__(self):
        self.config = DefaultConfig.audio_config

    def read_clean_data(self, file_path=""):
        """
        :return: 读入的干净音频序列及采样频率
        """
        if not file_path:
            return librosa.load(self.config.clean_data_path)
        else:
            return librosa.load(file_path)

    def read_noise_data(self, file_path=""):
        """
        如果配置中noise不为None，则读入纯噪声序列
        :return:
        """
        if not file_path:
            if self.config.noise_data_path is not None:
                data, fs = librosa.load(self.config.noise_data_path)
                return data, fs
            else:
                pass
        else:
            return librosa.load(file_path)

    def read_noisy_data(self, file_path=""):
        """
        如果config中的noisy不为None，则读入待处理的带噪序列
        :return:
        """
        if not file_path:
            if self.config.noisy_data_path is not None:
                data, fs = librosa.load(self.config.noisy_data_path)
                return data, fs
        else:
            return librosa.load(file_path)

    @staticmethod
    def write_data(data, fs, file_name="data.wav"):
        file_name = DefaultConfig.file_config.file_output_path + file_name
        sf.write(file_name, data, fs)


if __name__ == "__main__":
    audioUtils = AudioUtils()
    wave_fs, wave_data = audioUtils.read_clean_data()
    pass
