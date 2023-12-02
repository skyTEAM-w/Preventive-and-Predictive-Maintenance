import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tftb.processing
from pyhht import EMD
from scipy.signal import hilbert

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号


def picture(x, y, N):
    '''
    画信号的时域图和频谱
    输入：
    x: 0-1时间序列
    y: 信号
    N: 1s内采样点数
    输出：
    信号的时域图和频谱
    '''
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.xlabel('时间/s')
    plt.ylabel('幅值')
    plt.title('合成信号时域曲线')
    yf = np.fft.fft(y)
    xf = np.linspace(0.0, N / 2, N // 2)
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))  # 频谱幅值归一化，需要*2/N
    plt.xlabel('频率/Hz')
    plt.ylabel('幅值')
    plt.title('合成信号频谱')
    plt.show()


def HHT_Analysis(t, signal, N):
    decomposer = EMD(signal)
    imfs = decomposer.decompose()
    n_components = imfs.shape[0]
    fig1, axes = plt.subplots(n_components, 2, figsize=(10, 7), sharex='col', sharey=False)

    for i in range(n_components):
        axes[i][0].plot(t, imfs[i])
        axes[i][0].set_title('imf{}'.format(i + 1))

        yf = np.fft.fft(imfs[i])
        xf = np.linspace(0.0, N / 2, N // 2)
        axes[i][1].plot(xf, 2.0 / N * np.abs(yf[:N // 2]))  # 频谱幅值归一化，需要*2/N
        axes[i][1].set_title('IMF{}'.format(i + 1))
        pass
    plt.show()

    return imfs
    pass


def HHTPicture(t, imfs, N, n):
    '''
    画出指定个数的IMFs的时域图和时频图
    输入：
    t: 0-1时间序列
    imfs: IMFs成分
    N: 1s内采样点数
    n: 指定画前几个IMFs成分
    输出：
    前n个IMFs的时域图和时频图
    '''
    fig2, axes = plt.subplots(n, 2, figsize=(10, 7), sharex='col', sharey=False)
    # 计算并绘制各个组分
    for iter in range(n):
        # 绘制分解后的IMF时域图
        axes[iter][0].plot(t, imfs[iter])
        axes[iter][0].set_xlabel('时间/s')
        axes[iter][0].set_ylabel('幅值')
        # 计算各组分的Hilbert变换
        imfsHT = hilbert(imfs[iter])
        # 计算各组分Hilbert变换后的瞬时频率
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
        fs = N
        axes[iter][1].plot(timestamps / fs, instf * fs)
        axes[iter][1].set_xlabel('时间/s')
        axes[iter][1].set_ylabel('频率/Hz')
        # 计算瞬时频率的均值和中位数
        axes[iter][1].set_title(
            'Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))


def HHTFilter(signal, componentsRetain):
    '''
    定义HHT的滤波函数，提取部分EMD组分
    输入：
    signol: 信号
    componentsRetain: IMF的列表 []
    输出：
    一幅图，同时包含原始信号和合成信号
    '''
    # 进行EMD分解
    decomposer = EMD(signal)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 选取需要保留的EMD组分，并且将其合成信号
    signalRetain = np.sum(imfs[componentsRetain], axis=0)
    # 绘图
    plt.figure(figsize=(10, 7))
    # 绘制原始数据
    plt.plot(signal, label='RawData')
    # 绘制保留组分合成的数据
    plt.plot(signalRetain, label='HHTData')
    # 绘制标题
    plt.title('RawData-----HHTData')
    # 绘制图例
    plt.legend()
    plt.show()
    return signalRetain


if __name__ == '__main__':
    # 生成0-1时间序列，共2048个点
    N = 2048
    t = np.linspace(0, 1, N)
    # 生成信号
    signal = (2 + np.cos(8 * np.pi * t)) * np.cos(40 * np.pi * (t + 1) ** 2) + np.cos(
        20 * np.pi * t + 5 * np.sin(2 * np.pi * t))
    # 画出原始信号的时域图和频谱
    picture(t, signal, N)

    imfs = HHT_Analysis(t, signal, N)
    HHTPicture(t, imfs, N, 2)
    signalRetain = HHTFilter(signal, [0, 1])
