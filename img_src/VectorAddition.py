# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:52:01 2020

@author: Ben Vanderlei
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

options = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}
ax.arrow(0,0,1,3,fc='b',ec='b',**options)
ax.arrow(1,3,2,-1,fc='b',ec='b',**options)
ax.arrow(0,0,3,2,fc='r',ec='r',**options)

ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.set_aspect('equal')
ax.set_xticks(np.arange(0,5,step = 1))
ax.set_yticks(np.arange(0,5,step = 1))

ax.text(1,2,'$U_1$')
ax.text(2,3,'$U_2$')
ax.text(2,1,'$U_1+U_2$')


ax.grid(True,ls=':')
