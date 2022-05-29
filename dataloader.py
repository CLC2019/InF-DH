import torch
import os
import numpy as np
from numpy import vstack
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class INFdata(Dataset):
    def __init__(self, datapath, labelpath):
        super(INFdata, self).__init__()
        cir = np.load(datapath)
        pos = np.load(labelpath)
        self.cir = torch.tensor(cir)
        self.pos = torch.tensor(pos, dtype=torch.float32)


    def __len__(self):
        return len(self.cir)

    def __getitem__(self, item):
        return self.cir[item], self.pos[item]

def get_loader(config):
    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)


    trainset_dir = '../dataset/InF-DH/WLI_3_1001_InF_DH662_FR1_drop1/cir.npy'
    label_dir = '../dataset/InF-DH/WLI_3_1001_InF_DH662_FR1_drop1/pos.npy'
    train_dataset = INFdata(trainset_dir, label_dir)
    print('#trainDataset', len(train_dataset))
    #print('#testDataset', len(test_dataset))
    train_size = int(0.98 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(train_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(7))

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               num_workers=0,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=8,
                                              shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':
    class config:
        batch_size = 32
    get_loader(config=config)
    print("hello")