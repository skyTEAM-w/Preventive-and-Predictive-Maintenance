import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        N, C, L = x.size()
        z = x.view(N, -1)
        return z


class CWRUCNN(nn.Module):
    def __init__(self, kernel_num1=81, kernel_num2=27, kernel_size=55, pad=0, ms1=16, ms2=16):
        super(CWRUCNN, self).__init__()
        layer = [
            nn.Conv1d(1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(kernel_num2 * 8, 101)
        ]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)
