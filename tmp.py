#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import numpy as np
import Auxiliary as A
import numpy as np
import matplotlib.pyplot as plt


# def generate_poisson(m):
#     # print(m)
#     k = 0
#     p = np.exp(-m)
#     # print('p', p)
#     sum_poisson = p
#     y = np.random.random()
#     # print('y', y)
#     if y < p:
#         pass
#     else:
#         while y >= sum_poisson and p != 0:
#             p = m * p / (k + 1)
#             # print(p)
#             sum_poisson += p
#             # print('sum', sum_poisson)
#             k += 1
#         if y <= sum_poisson:
#             return k - 1
#
# m1 = []
# i = 0
#
# while i < 10000:
#     a = generate_poisson(1)
#     # print(a)
#     if a != None:
#         m1.append(a)
#         i += 1
#
# m = np.mean(m1)
# c = np.cov(m1)
# # fig = plt.figure(dpi=64, figsize=(16, 9))
#
#
# def A(i):
#     if i == 0 or i == 1:
#         return 1
#     else:
#         tmp = 1
#         for j in range(1, i + 1):
#             tmp = tmp * j
#         return tmp
#
#
# def left(m, j):
#         tmp = 0
#         for j in range(0, j + 1):
#             tmp = tmp + np.exp(-m) * pow(m, j) / A(j)
#         return tmp
#
#
# def right(m, j):
#     tmp = 0
#     for j in range(0, j + 2):
#         tmp = tmp + np.exp(-m) * pow(m, j) / A(j)
#     return tmp
#
#
# m = 1
# k = []
# i = 0
# y = np.random.random()
# k_count = 0
# while 1:
#     if i > 9999:
#         break
#     if y < left(m, k_count):
#         y = np.random.random()
#         k_count = 0
#     if left(m, k_count) <= y <= right(m, k_count):
#         k.append(k_count)
#         i += 1
#         print(i, y, left(m, k_count), right(m, k_count))
#         y = np.random.random()
#         k_count = 0
#         continue
#     else:
#         k_count += 1
# mean = np.mean(k)
# cov = np.cov(k)
#
#
# def test(a, b):
#     for i in range(a, b):
#         print(left(m, i), '~', right(m, i))
#
#
# mlist = os.listdir(path + '/matrix/')
# slist = os.listdir(path + '/s11/')
# s = []
# for i in slist:
#     temp = []
#     s11temp = open(path + '/s11/' + str(i))
#     line = s11temp.readline()
#     while line:
#         num = list(map(float, line.split()))
#         temp.append(num)
#         line = s11temp.readline()
#     s11temp.close()
#     temp = np.array(temp)
#     s11re = temp[:, 1]
#     s11im = temp[:, 2]
#     s11 = np.zeros([49, 1])
#     for j in range(49):
#         s11[j] = abs(complex(s11re[j], s11im[j]))
#     s11 = 20 * np.log10(s11)
#     if min(s11) > -8:
#         s.append(i)
#
# for i in s:
#     temp = i.split('-')
#     os.remove(path + '/s11/' + temp[0] + '-s11.txt')
#     os.remove(path + '/matrix/' + temp[0] + '-m.txt')
#     print(temp)
#
# a = A.mtest(path, 1)
# b = A.mtest(path, 2)
#
# from matplotlib.patches import Rectangle
#
# class Annotate(object):
#     def __init__(self):
#         self.ax = plt.gca()
#         self.rect = Rectangle((0,0), 1, 1)
#         self.x0 = None
#         self.y0 = None
#         self.x1 = None
#         self.y1 = None
#         self.ax.add_patch(self.rect)
#         self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
#     def on_press(self, event):
#         print('press')
#         self.x0 = event.xdata
#         self.y0 = event.ydata
#     def on_release(self, event):
#         print('release')
#         self.x1 = event.xdata
#         self.y1 = event.ydata
#         self.rect.set_width(self.x1 - self.x0)
#         self.rect.set_height(self.y1 - self.y0)
#         self.rect.set_xy((self.x0, self.y0))
#         self.ax.figure.canvas.draw()
#
# a = Annotate()
# plt.show()


path = 'C:/Users/wxadssxgn/Desktop/123'
s11path = path + '/s11/'
mpath = path + '/matrix/'
lpath = path + '/label/'
