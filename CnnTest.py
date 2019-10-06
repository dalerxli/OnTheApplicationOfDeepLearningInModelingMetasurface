#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt
import matplotlib.pyplot as plt
import CnnModel
import Auxiliary
import restestmodel

# num = 1223


def t(num):
    model = restestmodel.ResNet(restestmodel.Bottleneck, [3, 4, 23, 3])
    # model = CnnModel.CNN()
    # model.load_state_dict(torch.load('CnnParameters.pt'))
    model.load_state_dict(torch.load('Res.pt'))
    model.eval()
    path = 'G:/TianYuze/DataSets'
    # path = 'C:/Users/Antlab/Desktop/ppp'
    a = Auxiliary.mtest(path, num)
    a = torch.from_numpy(a)
    a = a.view(1, 1, 16, 16)
    a = a.float()
    b = model(Variable(a))
    b = b.data
    b = b.view(49, 1)
    b = b.numpy()
    b = 20 * np.log10(b)
    # print(b)
    fig = plt.figure(dpi=64, figsize=(16, 9))
    plt.xlabel('GHz', fontsize=32)
    plt.ylabel("dB", fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.ylim(-26, 1)
    plt.xlim(8, 13)
    x = np.linspace(8, 13, 49)
    plt.plot(x, b, color='red', linewidth=3)
    c = Auxiliary.shows11(path, num)
    plt.show()
