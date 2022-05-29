import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import posmodel
from utils import logger_configuration, plt_cdf
import torch
import torch.nn as nn
from config import fp_pos_config as config
from dataloader import get_loader
import torch.nn.functional as F


device = config.device

def train(model, train_loader, test_loader, config, criterion, optimizer, scheduler):
    logger = logger_configuration(config, save_log=True)

    data_len = len(train_loader.dataset)
    iteration = 0
    for epoch in range(config.epochs):
        model.train()
        for batch_idx, _data in enumerate(train_loader):
            cir, pos = _data
            optimizer.zero_grad()
            ##summary(model, spectrograms)   #显示参数
            output = model(cir)
            #prepos = return_pos2(output)#
            loss = criterion(output, pos)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            iteration += 1

            if iteration % config.print_step == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(cir), data_len,
                           100. * batch_idx / len(train_loader), loss.item()))

        #model.eval()
        if (epoch+1) % config.test_step == 0:
            test(model, test_loader, criterion, logger)

        if (epoch+1) % config.save_step == 0:
            torch.save(model.state_dict(), config.models + '/Ep{}.pth'.format(epoch+1))
            #test(model, test_loader, criterion, logger)
        '''
        if (epoch+1) % config.test_step == 0:
            positiontest(model, postestloader, criterion, logger=None)
        '''

def test(model, test_loader, criterion, logger=None):
    if logger is None:
        logger = logger_configuration(config, save_log=False)
    print('\nevaluating…')
    model.eval()

    test_loss = 0
    pretruesum = 0
    prealllabel = 0
    meter_error = []
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            cir, pos = _data
            output = model(cir)
            #prepos = return_pos(output)  #
            loss = criterion(output, pos)
            test_loss += loss.item() / len(test_loader)
            for i in range(len(output)):
                xd = output[i, 0] - pos[i, 0]
                yd = output[i, 1] - pos[i, 1]
                xyd = math.sqrt(xd * xd + yd * yd)
                meter_error.append(xyd)
    plt_cdf(meter_error)

    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    logger.info('Test set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == "__main__":
    train_loader, test_loader = get_loader(config)
    model = posmodel()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    scheduler = None
    '''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config.learning_rate,
                                                    steps_per_epoch=int(len(train_loader)),
                                                    epochs=config.epochs,
                                                    anneal_strategy='linear')
    '''
    model_path = './history/cir3/models/Ep20.pth'
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict, strict=False)
    #train(model, train_loader, test_loader, config, criterion, optimizer, scheduler)
    test(model, test_loader, criterion, logger=None)