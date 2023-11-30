import sys

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import matplotlib.pyplot as plt
from src.torchHHT import hht

data_dir = 'data/java/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dim = 2048


def load_data(file_name):
    print("Load Data File" + file_name)
    file_data = pd.read_csv(data_dir + file_name)
    signal_array = []
    temp_signal = file_data.iloc[:, :].values
    temp_signal = temp_signal.squeeze()
    # print(temp_signal.shape)
    fs = temp_signal.shape[0]
    signal_torch = torch.tensor(temp_signal, dtype=torch.float32)
    signal_torch.to(device)
    imfs, imfs_env, imfs_freq = hht.hilbert_huang(signal_torch, fs=fs, num_imf=10)
    imfs = imfs.cpu().numpy()
    imfs = np.sum(imfs[1:4], axis=0)
    # print(imfs.shape)
    sample_num = imfs.shape[0] // dim
    signal = imfs[0:dim * sample_num]
    signals = np.array(np.split(signal, sample_num))
    signal_array.append(signals)
    return np.concatenate(signal_array).squeeze()
    pass


# if __name__ == '__main__':
#     print(load_data('received_file.txt'))
