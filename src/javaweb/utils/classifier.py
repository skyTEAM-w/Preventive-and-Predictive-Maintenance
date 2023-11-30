import numpy as np
import torch
from tqdm import tqdm

from src.CWRU.config import Config
from src.CWRU.utils import create_model

opt = Config()


def classifier_model(signal_array=np.ndarray, model_dir=str):
    """
    对信号数组进行分类。

    Args:
        signal_array (numpy.ndarray): 信号数组。
        model_dir (str): 模型文件路径。

    Returns:
        int: 分类结果。
    """
    # 创建模型
    model = create_model(opt.model, opt.model_param)

    # 加载模型权重
    model.load_state_dict(torch.load(model_dir))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化分数记录
    score_hist = []

    # 设置模型为评估模式
    model.eval()

    # 对信号数组进行分类
    for i in tqdm(range(signal_array.shape[0]), desc='处理中', unit='次'):
        x = signal_array[i]
        tensor_x = torch.Tensor(x).unsqueeze(0).unsqueeze(1)
        tensor_x = tensor_x.float()
        tensor_x = tensor_x.to(device)

        score = model(tensor_x)
        score = score.max(1, keepdim=True)[1]
        score = score.item()
        score_hist.append(score)

    # 返回最终分类结果
    return np.argmax(np.bincount(np.array(score_hist)))
