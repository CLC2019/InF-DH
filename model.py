import torch.nn as nn
import torch
import torch.nn.functional as F

class rescnn(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel=3, stride=1):
        super(rescnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += res
        x = self.relu(x)
        return x


class posmodel(nn.Module):   #
    def __init__(self, n_cnnlayers=12):
        super(posmodel, self).__init__()
        self.cnn = nn.Conv2d(2, 32, 3, padding=1, stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.rescnn_layers = nn.Sequential(*[
            rescnn(32, 32, kernel=3, stride=1)
            for _ in range(n_cnnlayers)
        ])
        self.cnn2 = nn.Conv2d(32, 2, 3, padding=1, stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2304, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 2)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = self.bn1(x)
        x = self.rescnn_layers(x)
        x = self.cnn2(x)
        #x = self.bn2(x)
        x =self.flatten(x)
        #sizes = x.size()
        #x = x.reshape(sizes[0], -1)
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = self.bn4(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear3(x)
        return x