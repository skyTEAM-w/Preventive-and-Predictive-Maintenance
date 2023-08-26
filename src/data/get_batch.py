import numpy as np


def get_batch(train_x, train_y, time_step):
    length = len(train_x) - time_step
    sequence = []
    reserve = []

    for i in range(length):
        sequence.append(train_x[i:i + time_step])
        reserve.append(train_y[i:i + time_step])

    return np.array(sequence), np.array(reserve)
    pass
