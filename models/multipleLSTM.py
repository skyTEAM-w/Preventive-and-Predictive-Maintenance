import torch.nn as nn


class multipleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(multipleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x)
        # output = self.dropout(output)
        output = self.linear(lstm_output)
        # output = self.relu2(output)
        return output

    pass
