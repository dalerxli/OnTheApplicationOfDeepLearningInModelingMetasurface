#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np

from Tkinter import *


def drawboard(board, colors, startx=50, starty=50, cellwidth=50):
    width = 2 * startx + len(board) * cellwidth
    height = 2 * starty + len(board) * cellwidth
    canvas.config(width=width, height=height)
    for i in range(len(board)):
        for j in range(len(board)):
            index = board[i][j]
            color = colors[index]
            cellx = startx + i * 50
            celly = starty + j * 50
            canvas.create_rectangle(cellx, celly, cellx + cellwidth, celly + cellwidth,
                                    fill=color, outline="black")
    canvas.update()


root = Tk()
canvas = Canvas(root, bg="white")
canvas.pack()
board = [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'pink']
drawboard(board, colors)
root.mainloop()