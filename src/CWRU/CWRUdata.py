import h5py
import torch.utils.data as data


class CWRUdata(data.Dataset):

    def __init__(self, root, train):
        f = h5py.File(root, 'r')
        if train:
            self.X = f['X_train'][:]
            self.y = f['y_train'][:]
        else:
            self.X = f['X_test'][:]
            self.y = f['y_test'][:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
