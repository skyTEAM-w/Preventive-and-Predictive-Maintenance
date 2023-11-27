import numpy as np
import torch
import pandas as pd
from CWRUdata import CWRUdata


def check_accuracy(model, loader, device, error_analysis=False):
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        # one batch
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix


def create_dataset(data_dir, train):
    dataset = None
    if data_dir == 'DE1.h5' or data_dir == 'FE1.h5':
        dataset = CWRUdata(data_dir, train)
    else:
        raise ValueError("Dataset [%s] not recognized." % data_dir)
    print("dataset [%s] was created" % data_dir)
    return dataset


def create_model(model_name, model_param):
    model = None
    if model_name == 'plain_cnn':
        from CWRUCNN import CWRUCNN
        model = CWRUCNN(**model_param)
    else:
        raise NotImplementedError('model [%s] not implemented.' % model_name)
    print("model [%s] was created" % model_name)
    return model
