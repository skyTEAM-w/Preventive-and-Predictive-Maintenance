import numpy as np

from src.CWRU.config import Config
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from src.CWRU.utils import create_model
from tqdm import tqdm
import copy
import time
import pandas as pd
from torchsummary import summary
from src.CWRU.CWRUCNN import CWRUCNN

opt = Config()


def classifier_model(signal_array=np.ndarray, model_dir=str):
    model = create_model(opt.model, opt.model_param)

    model.load_state_dict(torch.load(model_dir))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # print(model)
    # print(signal_array.shape)
    score_hist = []
    model.eval()
    for i in tqdm(range(signal_array.shape[0]), desc='Processing', unit='iteration'):
        x = signal_array[i]
        tensor_x = torch.Tensor(x).unsqueeze(0).unsqueeze(1)
        tensor_x = tensor_x.float()
        tensor_x = tensor_x.to(device)

        score = model(tensor_x)
        score = score.max(1, keepdim=True)[1]
        score = score.item()
        # print(score)
        score_hist.append(score)
        # print()
        pass
        # print(signal_array[i].shape)
    return np.argmax(np.bincount(np.array(score_hist)))
    pass
