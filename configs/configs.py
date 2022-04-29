import os

# 音频读取的相关配置
AUDIO_CLEAN_DATA_FILE_PATH = "resources/sound_original.wav"
AUDIO_NOISE_DATA_FILE_PATH = "resources/noise.wav"
AUDIO_NOISY_DATA_FILE_PATH = "resources/noisy_sound.wav"

# 人工噪声的相关配置
NOISE_FREQUENCY_LOW = 2400
NOISE_FREQUENCY_HIGH = 3200
NOISE_SNR = 5

# 维纳滤波器的相关配置
# WIENER_N_FFT = 256                      # 傅里叶变换选择的序列长度
WIENER_N_FFT = 512                      # 傅里叶变换选择的序列长度
WIENER_ALPHA = 1                        # 维纳滤波的alpha参数
WIENER_BETA = 3                         # 维纳滤波的beta参数
# WIENER_FRAME_SHIFT_LENGTH = 128         # 维纳滤波时取信号的帧移长度
WIENER_FRAME_SHIFT_LENGTH = 256         # 维纳滤波时取信号的帧移长度
# WIENER_WINDOW_LENGTH = 256              # 维纳滤波时处理的信号窗长
WIENER_WINDOW_LENGTH = 512              # 维纳滤波时处理的信号窗长

# 文件相关配置
FILE_PATH = "./resources/"
FILE_OUTPUT_PATH = "./output/"


# 谱减法滤波的相关配置
# SUB_FILTER_N_FFT = 256                  # 傅里叶变换选择的序列长度
SUB_FILTER_N_FFT = 512                  # 傅里叶变换选择的序列长度
# SUB_FILTER_FRAME_SHIFT_LENGTH = 128     # 谱减法中的帧移长度
SUB_FILTER_FRAME_SHIFT_LENGTH = 256     # 谱减法中的帧移长度
# SUB_FILTER_WINDOW_LENGTH = 256          # 谱减法中处理的信号窗长
SUB_FILTER_WINDOW_LENGTH = 512          # 谱减法中处理的信号窗长
SUB_FILTER_ALPHA = 4                    # 谱减法中的alpha参数
SUB_FILTER_BETA = 0.0001                # 谱减法中的beta参数
SUB_FILTER_GAMMA = 1                    # 谱减法中的gamma参数
SUB_FILTER_K = 1                        # 谱减法中的k参数

