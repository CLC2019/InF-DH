from datetime import datetime
import torch

class fp_pos_config():
    CUDA = True
    device = torch.device("cpu")

    # logger
    print_step = 100
    test_step = 1
    #filename = datetime.now().__str__()[:-7]
    filename = 'cir4'
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    batch_size = 128
    epochs = 50
    save_step = 5
    learning_rate = 1e-4


#cir3 Ep20.pth可用
#model为12层