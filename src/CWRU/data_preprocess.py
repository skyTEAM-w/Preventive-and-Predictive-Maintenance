import h5py
import numpy as np
import os
import sys
import pandas as pd
import scipy.io as scio
import torch
import matplotlib.pyplot as plt
from src.torchHHT import hht
from src.hht.hht import signal_window

# %%
data_dir = '../../data/input/CWRU/raw_data/'

frame = pd.read_table('annotations.txt')

dim = 2048

train_fraction = 0.8

signal_train = []
label_train = []
signal_test = []
label_test = []

count = 0

# %%
for index in range(len(frame)):
    mat_name = os.path.join(data_dir, frame['file_name'][index])
    raw_data = scio.loadmat(mat_name)
    for key, data in raw_data.items():
        if key[5:7] == 'DE':
            signal = data.squeeze()
            temp_signal = signal
            # signal = signal_window(signal, 0.3)
            fs = signal.shape[0]
            if torch.cuda.is_available():
                signal = torch.Tensor(signal)
                signal = signal.cuda()
                pass
            imfs, imfs_env, imfs_freq = hht.hilbert_huang(signal, fs=fs, num_imf=10)
            imfs = imfs.cpu().numpy()
            imfs = np.sum(imfs[1:4], axis=0)
            # print(imfs)

            sample_num = imfs.shape[0] // dim

            train_num = int(sample_num * train_fraction)
            test_num = sample_num - train_num

            signal = imfs[0: dim * sample_num]

            signals = np.array(np.split(signal, sample_num))

            signal_train.append(signals[0: train_num])
            signal_test.append(signals[train_num: sample_num])

            label_train.append(index * np.ones(train_num))
            label_test.append(index * np.ones(test_num))

            # plt.figure(figsize=(20, 10))
            # plt.plot(temp_signal, label='origin')
            # plt.plot(imfs, label='imf')
            # plt.legend()

        pass
    pass

signals_tr_np = np.concatenate(signal_train).squeeze()
labels_tr_np = np.concatenate(np.array(label_train)).astype('uint8')
signals_tt_np = np.concatenate(signal_test).squeeze()
labels_tt_np = np.concatenate(np.array(label_test)).astype('uint8')
print(signals_tr_np.shape, labels_tr_np.shape, signals_tt_np.shape, labels_tt_np.shape)
# %%
signals_tr_np
# %%
labels_tr_np
# %%
f = h5py.File('DE1.h5', 'w')
f.create_dataset('X_train', data=signals_tr_np)
f.create_dataset('y_train', data=labels_tr_np)
f.create_dataset('X_test', data=signals_tt_np)
f.create_dataset('y_test', data=labels_tt_np)
f.close()
