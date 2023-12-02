class Config(object):
    train_dir = 'DE1.h5'  # train_dir required by customed dataset in data folder
    val_dir = 'DE1.h5'
    test_dir = 'DE1.h5'
    batch_size = 32  # batch size in dataloader

    model = 'plain_cnn'  # model selected, required by create_model in model.__init__

    epochs = 50
    lr = 0.001
    lr_decay_iters = 1
    lr_decay = 0.99

    print_every = 100  # print results interval

    device = 'cuda:0'
    model_param = {'kernel_num1': 54, 'kernel_num2': 54,
                   'kernel_size': 55, 'pad': 0, 'ms1': 16, 'ms2': 16}
    model_param['pad'] = int((model_param['kernel_size'] - 1) / 2)
