#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random


def GetMatrix(path, order):
    matrixpath = path + '/matrix/'
    matrix = np.zeros([len(order), 16, 16])
    t = 0
    for i in order:
        temp = []
        matrixtemp = open(matrixpath + str(i) + '-m.txt')
        line = matrixtemp.readline()
        while line:
            num = list(map(float, line.split()))
            temp.append(num)
            line = matrixtemp.readline()
        matrixtemp.close()
        matrix[t] = np.array(temp)
        t = t + 1
    matrix[matrix < 0.5] = 0
    matrix[matrix > 0.5] = 1
    return matrix


def GetS11(path, order):
    s11path = path + '/s11/'
    s11 = np.zeros([len(order), 49, 3])
    t = 0
    for i in order:
        temp = []
        s11temp = open(s11path + str(i) + '-s11.txt')
        line = s11temp.readline()
        while line:
            num = list(map(float, line.split()))
            temp.append(num)
            line = s11temp.readline()
        s11temp.close()
        s11[t] = np.array(temp)
        t = t + 1
    s11real = s11[:, :, 1]
    s11imaginary = s11[:, :, 2]
    return s11real, s11imaginary


def shows11(path, i):
    s11path = path + '/s11/'
    temp = []
    s11temp = open(s11path + str(i) + '-s11.txt')
    line = s11temp.readline()
    while line:
        num = list(map(float, line.split()))
        temp.append(num)
        line = s11temp.readline()
    s11temp.close()
    s11 = np.array(temp)
    s11real = s11[:, 1]
    s11imaginary = s11[:, 2]
    s11 = np.zeros([49, 1])
    for j in range(49):
        s11[j] = abs(complex(s11real[j], s11imaginary[j]))
    x = np.linspace(8, 13, 49)
    s11 = 20 * np.log10(s11)
    fig = plt.figure(dpi=64, figsize=(16, 9))
    plt.xlabel('GHz', fontsize=32)
    plt.ylabel("dB", fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.ylim(-26, 1)
    plt.xlim(8, 13)
    plt.plot(x, s11, color='green', linewidth=3)
    # plt.savefig(path + '/s11plot/' + str(i) + '.png')
    # plt.close(fig)
    plt.show()


def mtest(path, i):
    matrixpath = path + '/matrix/'
    matrix = []
    matrixtemp = open(matrixpath + str(i) + '-m.txt')
    line = matrixtemp.readline()
    while line:
        num = list(map(float, line.split()))
        matrix.append(num)
        line = matrixtemp.readline()
    matrixtemp.close()
    matrix = np.array(matrix)
    matrix[matrix < 0.5] = 0
    return matrix


def Shuffle(order, batch):
    order = random.sample(order, len(order))
    if len(order) / batch != 0:
        order = order * batch
    return order


def FileComparison(path, x, y):
    PathMatrix = path + '/matrix/'
    PathS11 = path + '/s11/'
    MatrixList = os.listdir(PathMatrix)
    S11List = os.listdir(PathS11)
    for i in range(x, y):
        mTemp = MatrixList[i]
        sTemp = S11List[i]
        mTemp = int(mTemp.split('-')[0])
        sTemp = int(sTemp.split('-')[0])
        if mTemp != sTemp:
            return i
        else:
            return -1


def MatrixComparison(path, obj_path):
    PathMatrix = path + '/matrix/'
    MatrixList = os.listdir(PathMatrix)
    out = []
    if obj_path == 0:
        for i in range(0, len(MatrixList) - 1):
            for j in range(i + 1, len(MatrixList)):
                MatrixTemp0 = open(PathMatrix + MatrixList[i])
                MatrixTemp1 = open(PathMatrix + MatrixList[j])
                temp0 = []
                temp1 = []
                line0 = MatrixTemp0.readline()
                while line0:
                    num0 = list(map(float, line0.split()))
                    temp0.append(num0)
                    line0 = MatrixTemp0.readline()
                MatrixTemp0.close()
                line1 = MatrixTemp1.readline()
                while line1:
                    num1 = list(map(float, line1.split()))
                    temp1.append(num1)
                    line1 = MatrixTemp1.readline()
                MatrixTemp1.close()
                temp0 = np.array(temp0)
                temp1 = np.array(temp1)
                if (temp0 == temp1).all():
                    out.append([MatrixList[i], MatrixList[j]])
                print('i=', i, 'j=', j)
                j += 1
            i += 1
        return out
    else:
        MatrixTempObj = open(obj_path)
        tempObj = []
        lineObj = MatrixTempObj.readline()
        while lineObj:
            numObj = list(map(float, lineObj.split()))
            tempObj.append(numObj)
            lineObj = MatrixTempObj.readline()
        MatrixTempObj.close()
        tempObj = np.array(tempObj)
        for i in range(0, len(MatrixList)):
            tempCmp = []
            MatrixTempCmp = open(PathMatrix + MatrixList[i])
            lineCmp = MatrixTempCmp.readline()
            while lineCmp:
                numCmp = list(map(float, lineCmp.split()))
                tempCmp.append(numCmp)
                lineCmp = MatrixTempCmp.readline()
            MatrixTempCmp.close()
            tempCmp = np.array(tempCmp)
            if (tempObj == tempCmp).all():
                out.append([MatrixList[i]])
        return out


def xxxxx(path, n, a):
    PathMatrix = path + '/matrix/'
    MatrixList = os.listdir(PathMatrix)
    epoch = 0
    count = 0
    k = 0
    while 1:
        k += 1
        print(k)
        tmp = np.zeros([16, 16])
        tmp0 = np.random.rand(8, 8)
        for i in range(0, tmp0.shape[0]):
            for j in range(0, tmp0.shape[1]):
                if tmp0[i][j] > a:
                    tmp0[i][j] = 1
                else:
                    tmp0[i][j] = 0
        tmp[0: 8, 0: 8] = tmp0
        tmp[0: 8, 8: 16] = tmp[0: 8, 7:: -1]
        tmp[8: 16, ] = tmp[7::-1, ]
        for i in range(0, len(MatrixList)):
            tempCmp = []
            MatrixTempCmp = open(PathMatrix + MatrixList[i])
            lineCmp = MatrixTempCmp.readline()
            while lineCmp:
                numCmp = list(map(float, lineCmp.split()))
                tempCmp.append(numCmp)
                lineCmp = MatrixTempCmp.readline()
            MatrixTempCmp.close()
            tempCmp = np.array(tempCmp)
            if (tmp != tempCmp).all():
                count += 1
        if count == len(MatrixList):
            epoch += 1
            np.savetxt('file_name.txt', tmp)
            count = 0
            break
        # if epoch > n - 1:
            # break


def MatrixGenerate(a):
    tmp = np.zeros([16, 16])
    tmp0 = np.random.rand(8, 8)
    for i in range(0, tmp0.shape[0]):
        for j in range(0, tmp0.shape[1]):
            if tmp0[i][j] > a:
                tmp0[i][j] = 1
            else:
                tmp0[i][j] = 0
    tmp[0: 8, 0: 8] = tmp0
    tmp[0: 8, 8: 16] = tmp[0: 8, 7:: -1]
    tmp[8: 16, ] = tmp[7::-1, ]
    return tmp


def mFileGenerate(path, n, a, label):
    PathMatrix = path + '/matrix/'
    count = 0
    while 1:
        tmp = MatrixGenerate(a)
        pathTemp = PathMatrix + str(label) + '-m.txt'
        np.savetxt(pathTemp, tmp)
        resCmp = MatrixComparison(path, pathTemp)
        if len(resCmp) > 1:
            os.remove(pathTemp)
        else:
            count += 1
            print(label)
            label += 1
        if count > n - 1:
            break


def RepeatRemove(path):
    PathMatrix = path + '/matrix/'
    MatrixList = os.listdir(PathMatrix)
    sets = np.zeros([len(MatrixList), 16, 16])
    re = []
    for i in range(0, len(MatrixList)):
        MatrixTemp = open(PathMatrix + MatrixList[i])
        temp = []
        line = MatrixTemp.readline()
        while line:
            num = list(map(float, line.split()))
            temp.append(num)
            line = MatrixTemp.readline()
        MatrixTemp.close()
        sets[i] = np.array(temp)
    for i in range(0, len(MatrixList)):
        for j in range(i + 1, len(MatrixList)):
            if (sets[i] == sets[j]).all():
                re.append([MatrixList[i], MatrixList[j]])
                print('i=', i, 'j=', j)
    if re == []:
        print('no repeating items')
    else:
        res = np.array(re)
        res2 = res[:, 1]
        namespace = []
        for i in range(0, len(res2)):
            namespace.append(res2[i].split('-')[0])
        for i in range(0, len(namespace)):
            mpath = path + '/matrix/' + namespace[i] + '-m.txt'
            spath = path + '/s11/' + namespace[i] + '-s11.txt'
            if os.path.exists(mpath) and os.path.exists(spath):
                os.remove(mpath)
                os.remove(spath)
                print('remove' + namespace[i] + '-m.txt')
            else:
                continue
        print('clear all repeating items')


def rename(path, startorder):
    PathMatrix = path + '/matrix/'
    MatrixList = os.listdir(PathMatrix)
    PathS11 = path + '/s11/'
    S11List = os.listdir(PathS11)
    namespace = []
    for i in range(0, len(MatrixList)):
        namespace.append(MatrixList[i].split('-')[0])
    for i in range(0, len(namespace)):
        new_m = PathMatrix + str(i + startorder) + '-m.txt'
        new_s = PathS11 + str(i + startorder) + '-s11.txt'
        old_m = PathMatrix + namespace[i] + '-m.txt'
        old_s = PathS11 + namespace[i] + '-s11.txt'
        os.rename(old_m, new_m)
        os.rename(old_s, new_s)
        print('rename', namespace[i], 'to', i + startorder)


def mGenerate(a):
    while 1:
        tmp = np.zeros([16, 16])
        tmp0 = np.random.rand(8, 8)
        for i in range(0, tmp0.shape[0]):
            for j in range(0, tmp0.shape[1]):
                if tmp0[i][j] > a:
                    tmp0[i][j] = 1
                else:
                    tmp0[i][j] = 0
        tmp[0: 8, 0: 8] = tmp0
        tmp[0: 8, 8: 16] = tmp[0: 8, 7:: -1]
        tmp[8: 16, ] = tmp[7::-1, ]
        count = str(tmp).count('1')
        print(count)
        if count == 32 * 4:
            print(count)
            break
    return tmp


def mfGenerate(path, n, a, label):
    path0 = 'G:/TianYuze/DataSets'
    PathMatrix = path + '/matrix/'
    count = 0
    while 1:
        tmp = mGenerate(a)
        pathTemp = PathMatrix + str(label) + '-m.txt'
        np.savetxt(pathTemp, tmp)
        resCmp = MatrixComparison(path0, pathTemp)
        if len(resCmp) > 1:
            os.remove(pathTemp)
        else:
            count += 1
            print('label:', label)
            label += 1
        if count > n - 1:
            break


def ssim(x, y, c1, c2):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    xy_cov = np.mean(np.cov(x, y))
    t1 = (2 * x_mean * y_mean + c1) * (2 * xy_cov + c2)
    t2 = (np.power(x_mean, 2) + np.power(y_mean, 2) + c1) * (np.power(x_std, 2) + np.power(y_std, 2) + c2)
    return t1 / t2


def findminimums(data):
    ans = []
    for i in range(len(data)):
        if i == 0:
            if data[i] < data[i + 1]:
                ans.append(i)
            else:
                continue
        if i == len(data) - 1:
            if data[i] < data[i - 1]:
                ans.append(i)
            else:
                continue
        if 0 < i < len(data) - 2:
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                ans.append(i)
    return ans


def makelabels(path):
    s11path = path + '/s11/'
    for i in range(1, 7803):
        temp = []
        s11temp = open(s11path + str(i) + '-s11.txt')
        line = s11temp.readline()
        while line:
            num = list(map(float, line.split()))
            temp.append(num)
            line = s11temp.readline()
        s11temp.close()
        s11 = np.array(temp)
        s11real = s11[:, 1]
        s11imaginary = s11[:, 2]
        s11 = np.zeros([49, 1])
        for j in range(49):
            s11[j] = abs(complex(s11real[j], s11imaginary[j]))
        s11 = 20 * np.log10(s11)
        label = np.zeros([49, 1])
        minlocation = findminimums(s11)
        for k in minlocation:
            if s11[k] < -15:
                label[k] = 1
        labelpath = path + '/label/'
        np.savetxt(labelpath + str(i) + '-location.txt', label)
        print('current:', i)


def getlabels(batch, path, order):
    labelpath = path + '/label/'
    label = np.zeros([batch, 49, 1])
    t = 0
    for i in order:
        temp = []
        labeltemp = open(labelpath + str(i) + '-location.txt')
        line = labeltemp.readline()
        while line:
            num = list(map(float, line.split()))
            temp.append(num)
            line = labeltemp.readline()
        labeltemp.close()
        label[t] = np.array(temp)
        t = t + 1
    label[label > 0.5] = 1
    label[label < 0.5] = -0.1
    return label
