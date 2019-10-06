#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt
import random
import CnnModel
import Auxiliary
import adabound
import restestmodel


dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor
# model = CnnModel.CNN().cuda()
model = restestmodel.ResNet(restestmodel.Bottleneck, [2, 2, 2, 2]).cuda()
# model.load_state_dict(torch.load('CnnParameters.pt'))

path = 'C:/Users/wxadssxgn/Desktop/123'
batch = 20

DataSets = list(range(1, 7803))
DataSetsOrders = Auxiliary.Shuffle(DataSets, batch)
# criterion = nn.MSELoss(reduction='sum')
criterion = nn.CrossEntropyLoss()

# optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=1e-5)
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)

# epoch = 0
loop = 0
for epoch in range(1000):
    # while 1:
    time = dt.datetime.now().isoformat()
    order = DataSetsOrders[loop: loop + batch]
    if loop == int(len(DataSetsOrders) / batch):
        loop = 0
    loop += batch
    epoch += 1
    # order = random.sample(DataSetsOrders, batch)
    a = Auxiliary.GetMatrix(path, order)
    # d = np.zeros([batch, 49, 1])
    # b, c = Auxiliary.GetS11(path, order)
    # for j in range(batch):
    #     for i in range(np.size(b[0])):
    #         temp = abs(complex(b[j][i], c[j][i]))
    #         d[j][i] = temp
    # d = torch.from_numpy(d)
    # d = d.float()
    # d = d.view(batch, 49)
    d = Auxiliary.getlabels(batch, path, order)
    d = torch.from_numpy(d)
    d = d.float()
    d = d.view(batch, 49)
    a = torch.from_numpy(a)
    a = a.float()
    a = a.view(batch, 1, 16, 16)
    x = Variable(a.type(dtype), requires_grad=False)
    y = Variable(d.type(dtype), requires_grad=False)
    netout = model(x)
    loss = criterion(netout, y)
    if epoch % 1 == 0:
        print(time, epoch, loss.data)
    if loss.data < 1e-2:
        print(time, epoch, loss.data)
        break
    # if epoch % 100 == 0:
    # torch.save(model.state_dict(), 'CnnParameters.pt')
    # model.load_state_dict(torch.load('CnnParameters.pt'))
    # print(time, epoch, loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5000 == 0:
        DataSetsOrders = Auxiliary.Shuffle(DataSets, batch)
        loop = 0

# torch.save(model.state_dict(), 'CnnParameters.pt')
