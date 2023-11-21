from src.data.data_loader import load_data_cuda
from src.data.data_preprocessing import preprocessor
from src.data.get_batch import get_batch
from models.multipleLSTM import multipleLSTM
from src.train.train_model import train
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn

if __name__ == '__main__':
    data_dir = '../data/input/NASA/3rd_test/4th_test/txt/'
    data_col = ['b1', 'b2', 'b3', 'b4']
    df = load_data_cuda(data_dir, data_col)
    print(df.head())
    df.to_csv('../data/input/NASA/3rd_test/4th_test/txt/4th_test.csv', index=False)
    df = pd.read_csv('../data/output/4st_test_energy.csv', index_col=0, header=0)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    df.columns = data_col
    df.plot()
    plt.title('4th test data')
    plt.show()
    pass