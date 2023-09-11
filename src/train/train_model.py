import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import numpy as np


def train(model=nn.Module, dataloader=torch.utils.data.DataLoader, test_dataloader=None, epochs=int):
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer_fn = optim.Adam(params=model.parameters(), lr=1e-3)
    reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_fn,
                                                  mode='min', factor=0.1, patience=10,
                                                  verbose=False, threshold=1e-4, threshold_mode='rel',
                                                  cooldown=0, min_lr=0, eps=1e-8)

    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        model = model.cuda()

    train_hist = np.zeros(epochs)
    test_hist = np.zeros(epochs)

    for i in range(epochs):

        for train_data, train_label in dataloader:

            pre = model(train_data)
            tr_loss = loss_fn(pre.float(), train_label)
            tr_loss.requires_grad_(True)

            if test_dataloader is not None:
                with torch.no_grad():
                    for test_data, test_label in test_dataloader:
                        pre_test = model(test_data)
                        test_loss = loss_fn(pre_test.float(), test_label)
                        pass
                test_hist[i] = test_loss.item()
                if i % 10 == 0:
                    print(f'Epoch {i} train loss: {tr_loss.item()} test loss: {test_loss.item()}')
                    pass
                pass

            elif i % 10 == 0:
                print(f'Epoch {i} train loss: {tr_loss.item()}')
                pass

            train_hist[i] = tr_loss.item()
            optimizer_fn.zero_grad()
            tr_loss.backward()
            optimizer_fn.step()
            reduce.step(tr_loss)
            pass

        pass

        pass

    return model.eval(), train_hist, test_hist

    pass
