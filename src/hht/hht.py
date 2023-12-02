import matplotlib.pyplot as plt
import numpy as np
import scipy
import tftb
import tftb.processing
from pyhht import EMD
from scipy import signal as scisignal
from scipy.integrate import simps


def signal_window(signal, beta=float) -> np.ndarray:
    """
    信号加窗，抑制HHT边缘效应
    :param signal: 输入信号
    :param beta: beta
                Window shape
                0
                Rectangular
                5
                Similar to a Hamming
                6
                Similar to a Hanning
                8.6
                Similar to a Blackman
    :return: 加窗后信号
    """
    signal_temp = scipy.signal.detrend(signal, type='linear')
    return signal_temp * np.kaiser(len(signal_temp), beta)
    pass


def hht_analysis(signal, time, size):
    """
    HHT分析
    :param signal: 输入信号
    :param time: 时间
    :param size: 序列大小
    :return:
    """
    decomposer = EMD(signal)
    imfs = np.array(decomposer.decompose())
    n_component = imfs.shape[0]
    imfsht = []
    for i in range(n_component):
        imfs_ht_temp = scisignal.hilbert(imfs[i])
        imfsht.append(imfs_ht_temp)
        pass
    imfs_ht = np.array(imfsht)
    return imfs, imfs_ht
    pass


def hht_picture(imfs, imfs_ht, time, size):
    n_component = imfs.shape[0]
    fig, axes = plt.subplots(n_component, 3, figsize=(20, 30), sharex='col', sharey=False, linewidth=2)
    for i in range(n_component):
        axes[i][0].plot(time, imfs[i])
        axes[i][0].set_title('imf{}'.format(i + 1))

        fft = np.fft.fft(imfs[i])
        x_fft = np.linspace(0.0, size / 2, size // 2)
        axes[i][1].plot(x_fft, 2.0 / size * np.abs(fft[:size // 2]))
        axes[i][1].set_title('IMF{}---FFT'.format(i + 1))

        fs = size
        imfs_ht_temp = imfs_ht[i][:, np.newaxis]
        instf, timestamps = tftb.processing.inst_freq(imfs_ht_temp)
        axes[i][2].plot(timestamps / fs, instf * fs)
        axes[i][2].set_ylabel('Frequency/Hz')
        axes[i][2].set_xlabel('Time/s')
        axes[i][2].set_title('IMF{}---Frequency/s'.format(i + 1))
        pass
    plt.tight_layout()
    plt.show()
    pass


def feature_processing_inst_freq(imfs_ht=np.ndarray) -> np.ndarray:
    n_component = imfs_ht.shape[0]
    frequency_array = []
    for i in range(n_component):
        fs = 20480
        imfs_ht_temp = imfs_ht[i][:, np.newaxis]
        instf, timestamps = tftb.processing.inst_freq(imfs_ht_temp)
        frequency_array.append(np.mean(instf * fs))
        pass
    return np.array(frequency_array)
    pass


def feature_processing_energy(imfs=np.ndarray, imfs_ht=np.ndarray, fs=int, size=int) -> float:
    n_component = imfs.shape[0]
    imfs_ht = imfs_ht[:-1]
    energy = 0.

    for i in range(imfs_ht.shape[0]):
        analytic_signal = imfs_ht[i]
        amp = np.abs(analytic_signal)
        energy_temp = simps(amp ** 2, dx=1. / fs)
        energy += energy_temp
        pass
    return energy
    pass

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert
# from PyEMD import EMD
#
# # 生成示例信号（可以替换为你自己的信号数据）
# sample_rate = 20480
# time = np.linspace(0, 1, sample_rate)
# frequency1 = 5
# frequency2 = 50
# signal = np.sin(2 * np.pi * frequency1 * time) + np.sin(2 * np.pi * frequency2 * time)
#
# # 进行经验模态分解（EMD）
# emd = EMD()
# imf = emd(signal)
# imf = imf[:-1]  # 去除最后一个 imf
#
# # 初始化时间频率幅度谱
# time_freq_amp_spectrum = []
#
# # 遍历每个 IMF
# for i in range(len(imf)):
#     analytic_signal = hilbert(imf[i])  # 计算解析信号
#     instantaneous_frequency = np.diff(np.unwrap(np.angle(analytic_signal)))  # 计算瞬时频率
#
#     time_points = np.arange(len(imf[i]))  # 时间点
#     freq_points = instantaneous_frequency * (sample_rate / (2 * np.pi))  # 瞬时频率转换为实际频率
#
#     # 将瞬时频率与时间点拼接起来，作为时间频率幅度谱的一部分
#     time_freq_amp_spectrum.append(np.vstack((time_points, freq_points)))
#
# # 将时间频率幅度谱的各个部分合并为一个二维数组
# time_freq_amp_spectrum = np.array(time_freq_amp_spectrum)
#
# # 绘制时间频率幅度谱
# plt.pcolormesh(time_freq_amp_spectrum[:, 0, :], time_freq_amp_spectrum[:, 1, :], np.abs(imf))
# plt.ylabel('Frequency')
# plt.xlabel('Time')
# plt.title('Time-Frequency Amplitude Spectrum')
# plt.colorbar(label='Amplitude')
# plt.show()
