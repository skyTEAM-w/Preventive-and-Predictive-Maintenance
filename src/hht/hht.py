import tftb.processing
from pyhht import EMD
from scipy import signal as scisignal
import pandas as pd
import numpy as np
from tftb import processing
import matplotlib.pyplot as plt


def signal_window(signal, beta):
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
    return signal * np.kaiser(len(signal), beta)
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
    fig, axes = plt.subplots(n_component, 3, figsize=(20, 30), sharex='col', sharey=False)
    for i in range(n_component):
        axes[i][0].plot(time, imfs[i])
        axes[i][0].set_title('imf{}'.format(i + 1))

        fft = np.fft.fft(imfs[i])
        x_fft = np.linspace(0.0, size / 2, size // 2)
        axes[i][1].plot(x_fft, 2.0 / size * np.abs(fft[:size // 2]))
        axes[i][1].set_title('IMF{}---FFT'.format(i + 1))

        fs = size
        instf, timestamps = tftb.processing.inst_freq(imfs_ht)
        axes[i][2].plot(timestamps / fs, instf * fs)
        axes[i][2].set_ylabel('Frequency/Hz')
        axes[i][2].set_xlabel('Time/s')
        axes[i][2].set_title('IMF{}---Frequency/s')
        pass
    plt.tight_layout()
    plt.show()
    pass

def feature_processing():

    pass
