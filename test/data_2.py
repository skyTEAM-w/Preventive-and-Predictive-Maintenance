from src.data.data_loader import load_data
from src.data.data_preprocessing import preprocessor
from src.data.get_batch import get_batch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn


class mutipleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(mutipleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output)
        return output

    pass


model = mutipleLSTM(input_size=4, output_size=4, num_layers=4, hidden_size=512, dropout=0.5)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('../models/lstm.pth'))
print(model)

if __name__ == '__main__':
    # data_dir = '../data/input/NASA/2nd_test/2nd_test/'
    data_col = ['b1', 'b2', 'b3', 'b4']
    # df = load_data(data_dir, data_col)
    # df.to_csv('../data/output/2nd_test_energy.csv')

    df = pd.read_csv('../data/output/2nd_test_energy.csv', index_col=0, header=0)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    df.columns = data_col
    plt.subplot(1, 2, 1)
    df.plot()
    plt.title('2nd_test')
    plt.show()

    pro = preprocessor(data=df, data_col=data_col)

    ddf = pro.data_preprocessing()

    ddf.plot()
    plt.title('2nd_test after PCA')
    plt.show()

    model.eval()

    data_test = ddf.to_numpy()

    test_x, test_y = get_batch(data_test[:-3], data_test[1:][:, 0:4 + 1], 4)

    test_x = torch.Tensor(test_x)
    test_x = test_x.cuda()
    test_x = test_x.float()

    y_out = model(test_x)
    y_out = y_out.cpu()
    y_out = y_out.detach().numpy()

    y_p = []
    y_t = []

    for i in range(y_out.shape[0]):
        y_m = np.mean(y_out[i, :, :], axis=0)
        y_p.append(y_m)

        y_true = np.mean(test_y[i, :, :], axis=0)
        y_t.append(y_true)

    y_pre = np.array(y_p).reshape(-1, 4)
    y_true = np.array(y_t).reshape(-1, 4)

    plt.figure(figsize=(20, 10))

    for i, index in enumerate(ddf.columns):
        ax1 = plt.subplot(4, 1, i + 1)
        plt.sca(ax1)
        plt.plot(y_pre[:, i], label="Pre-" + str(index), c='red')
        plt.plot(y_true[:, i], label="True-" + str(index), c='navy')
        plt.legend()
    plt.tight_layout()
    plt.show()
