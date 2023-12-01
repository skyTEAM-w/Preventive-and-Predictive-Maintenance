import numpy as np
import pandas as pd
import torch

from src.torchHHT import hht

data_dir = 'data/java/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dim = 2048


def load_data(file_name):
    """
    加载数据文件，进行HHT分析并提取信号。

    Args:
        file_name (str): 文件名。

    Returns:
        numpy.ndarray: 处理后的信号数组。
    """
    print("加载数据文件：" + file_name)

    # 读取CSV文件数据
    file_data = pd.read_csv(data_dir + file_name)
    signal_array = []

    # 提取信号数据
    temp_signal = file_data.iloc[:, :].values
    temp_signal = temp_signal.squeeze()
    fs = temp_signal.shape[0]

    # 转换为PyTorch张量并移动到设备上
    signal_torch = torch.tensor(temp_signal, dtype=torch.float32)
    signal_torch.to(device)

    # 进行HHT分析
    imfs, imfs_env, imfs_freq = hht.hilbert_huang(signal_torch, fs=fs, num_imf=10)
    imfs = imfs.cpu().numpy()
    imfs = np.sum(imfs[1:4], axis=0)

    # 提取部分信号
    sample_num = imfs.shape[0] // dim
    signal = imfs[0:dim * sample_num]
    signals = np.array(np.split(signal, sample_num))
    signal_array.append(signals)

    # 返回处理后的信号数组
    return np.concatenate(signal_array).squeeze()
