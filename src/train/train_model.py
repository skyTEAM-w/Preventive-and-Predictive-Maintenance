import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils


def train(model=nn.Module, dataloader=torch.utils.data.DataLoader, test_dataloader=None, epochs=int):
    # 定义损失函数
    loss_fn = nn.MSELoss(reduction='sum')
    # 定义优化器
    optimizer_fn = optim.Adam(params=model.parameters(), lr=1e-3)
    # 定义学习率调整策略
    reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_fn,
                                                  mode='min', factor=0.1, patience=10,
                                                  verbose=False, threshold=1e-4, threshold_mode='rel',
                                                  cooldown=0, min_lr=0, eps=1e-8)

    # 如果GPU可用，将损失函数和模型转换到GPU
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        model = model.cuda()

    # 初始化训练和测试历史记录
    train_hist = np.zeros(epochs)
    test_hist = np.zeros(epochs)

    # 开始训练
    for i in range(epochs):

        # 遍历训练数据
        for train_data, train_label in dataloader:

            # 计算模型输出
            pre = model(train_data)
            # 计算损失
            tr_loss = loss_fn(pre.float(), train_label)
            # 梯度求导
            tr_loss.requires_grad_(True)

            # 如果测试数据可用，计算测试损失
            if test_dataloader is not None:
                with torch.no_grad():
                    for test_data, test_label in test_dataloader:
                        pre_test = model(test_data)
                        test_loss = loss_fn(pre_test.float(), test_label)
                        pass
                test_hist[i] = test_loss.item()
                # 每10次训练输出一次训练和测试损失
                if i % 10 == 0:
                    print(f'Epoch {i} train loss: {tr_loss.item()} test loss: {test_loss.item()}')
                    pass
                pass

            # 每10次训练输出一次训练损失
            elif i % 10 == 0:
                print(f'Epoch {i} train loss: {tr_loss.item()}')
                pass

            # 记录训练损失
            train_hist[i] = tr_loss.item()
            # 梯度归零
            optimizer_fn.zero_grad()
            # 反向传播
            tr_loss.backward()
            # 更新参数
            optimizer_fn.step()
            # 根据损失调整学习率
            reduce.step(tr_loss)
            pass

        pass

        pass

    # 返回模型和训练和测试历史记录
    return model.eval(), train_hist, test_hist

    pass
