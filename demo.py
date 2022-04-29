import matplotlib.pyplot as plt
from NoiseUtil import Noise
from AudioUtil import AudioUtils
from WienerUtil import Wiener
from SubFilterUtil import SubFilterUtil


# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
TOTAL_PLOTS = 4

fig = plt.figure(figsize=(6, 8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)


def counter():
    num = 0
    while True:
        num += 1
        yield num


count_generator = counter()


def wiener_subproc_process(data_to_process, data_fs):
    """
    :param sound_data: sound file with noise
    :return:
    """
    wiener_sub_proc = SubFilterUtil()
    filtered_data = wiener_sub_proc.sub_proc(data_to_process)
    noise_data = data_to_process[:len(filtered_data)] - filtered_data
    wiener = Wiener()
    wiener.genFilter(noise_data, filtered_data)
    output_data = wiener.waveFiltering(data_to_process)
    AudioUtils.write_data(output_data, data_fs, "demo_wiener_subproc_filtered_data.wav")
    draw_plot(filtered_data, 512, data_fs, "超减法+维纳滤波")


def wiener_process(pure_noise, data_to_process, clean_data, data_fs):
    wiener = Wiener()
    wiener.genFilter(pure_noise, clean_data)
    filtered_data = wiener.waveFiltering(data_to_process)
    filtered_data *= 5
    AudioUtils.write_data(filtered_data, data_fs, "demo_wiener_filtered_data.wav")
    draw_plot(filtered_data, 512, data_fs, "维纳滤波")


def draw_plot(pic_data, para_NFFT, para_Fs, title, x_label="时间/s", y_label="频率/Hz"):
    plt.subplot(TOTAL_PLOTS, 1, next(count_generator))
    plt.specgram(pic_data, NFFT=para_NFFT, Fs=para_Fs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


if __name__ == "__main__":
    # 读入干净的语音
    audio_util = AudioUtils()
    data, fs = audio_util.read_clean_data("./resources/demo_sound_original.wav")

    # 生成噪声
    noise_util = Noise()
    noise_util.generateGaussianNoise(N=len(data), order_filter=256, fs=fs)
    noisy_data = noise_util.addGaussianNoise(data=data)
    AudioUtils.write_data(noisy_data, fs, "demo_noisy_data.wav")

    plt.title("音频频谱滤波结果")
    draw_plot(data, 512, fs, "原始无噪音频")
    draw_plot(noisy_data, 512, fs, "加噪后音频")

    # 原始维纳滤波处理
    wiener_process(noise_util.noise, noisy_data, data, data_fs=fs)

    # 使用了超减法后维纳滤波
    wiener_subproc_process(noisy_data, data_fs=fs)

    plt.show()
