# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:14:33 2020

@author: Ben Vanderlei
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

options = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}
ax.arrow(0,0,1,3,fc='b',ec='b',**options)
ax.arrow(2,0,2,6,fc='r',ec='r',**options)


ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_aspect('equal')
ax.set_xticks(np.arange(0,7,step = 1))
ax.set_yticks(np.arange(0,7,step = 1))

ax.text(1,2,'$U_1$')
ax.text(4,5,'$2U_1$')


ax.grid(True,ls=':')
