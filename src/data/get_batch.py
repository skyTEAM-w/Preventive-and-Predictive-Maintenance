import numpy as np


def get_batch(train_x, train_y, time_step):
    """
    LSTM数据集划分方法
    :param train_x: X
    :param train_y: Y
    :param time_step: 时间步长
    :return: LSTM训练集
    """
    length = len(train_x) - time_step
    sequence = []
    reserve = []

    for i in range(length):
        sequence.append(train_x[i:i + time_step])
        reserve.append(train_y[i:i + time_step])

    return np.array(sequence), np.array(reserve)
    pass


def get_cnn_batch(train_x, train_y):
    """
    CNN数据集划分方法
    :param train_x: X
    :param train_y: Y
    :return: CNN训练集
    """
    length = len(train_x) - 1
    sequence = []
    reserve = []

    for i in range(length):
        sequence.append(train_x[i])
        reserve.append(train_y[i])

    return np.array(sequence), np.array(reserve)
    pass
