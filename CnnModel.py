#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1, padding=0))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1, padding=0))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1, padding=0))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pool4 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1, padding=0))
        self.fc1 = nn.Sequential(nn.Linear(128 * 12 * 12, 2048), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(2048, 512), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(512, 49))
        self.softmax = nn.Sequential(nn.Softmax())

    def forward(self, temp):
        temp = self.conv1(temp)
        temp = self.pool1(temp)
        temp = self.conv2(temp)
        temp = self.pool2(temp)
        temp = self.conv3(temp)
        temp = self.pool3(temp)
        temp = self.conv4(temp)
        temp = self.pool4(temp)
        temp = temp.view(temp.size(0), -1)
        temp = self.fc1(temp)
        temp = self.fc2(temp)
        temp = self.out(temp)
        temp = self.softmax(temp)
        return temp
        # return F.softmax(self.out(temp), dim=1)
