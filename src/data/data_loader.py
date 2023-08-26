import pandas as pd
import numpy as np
import os
from src.hht.hht import signal_window, hht_analysis, feature_processing_energy


def load_data(data_dir=str, data_col=list):
    energy_data = pd.DataFrame()
    n_file = len(os.listdir(data_dir))
    cnt = 0

    for filename in os.listdir(data_dir):
        cnt += 1
        data_set = pd.read_csv(os.path.join(data_dir, filename), sep='\t', names=data_col)
        print(f'{cnt} / {n_file}')
        row_data = []

        for col_name in data_col:
            print(data_set[col_name].to_numpy())
            temp_signal = signal_window(signal=data_set[col_name], beta=3)
            _, imfs_ht = hht_analysis(signal=temp_signal, time=data_set.shape[0], size=data_set.shape[0])
            row_data.append(feature_processing_energy(imfs=imfs_ht, imfs_ht=imfs_ht, fs=data_set.shape[0], size=data_set.shape[0]))
        pass

        data_row = pd.DataFrame(np.array(row_data).reshape(1, data_set.shape[1]))
        data_row.index = [filename]
        energy_data = energy_data.append(data_row)
        pass

    return energy_data
    pass

