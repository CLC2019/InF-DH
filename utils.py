import logging
import os
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("DeepSC")
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s %(filename)s: %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def plt_cdf(data):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(0, max(data))
    y = ecdf(x)
    #设置标题，x，y标题
    plt.title("CDF")
    plt.xlabel("error m")
    plt.ylabel("probability")
    #设置坐标刻度
    yscale = np.arange(0, 1.1, 0.1)
    plt.yticks(yscale)
    xscale = np.arange(0, max(data), 1)
    plt.xticks(xscale)


    plt.plot(x, y)
    for a, b in zip(x, y):  # 添加这个循环显示坐标
        if b - 0.9 < 0.01 and b - 0.9 > 0:
            plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    #plt.show()
    plt.grid()
    plt.savefig("1.jpg")