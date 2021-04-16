# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:16:47 2021

@author: Ben Vanderlei
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

coords = np.array([[0,0],[1.5,0.5],[1.5,1.5],[0,1],[0,0]])
coords = coords.transpose()

x = coords[0,:]
y = coords[1,:]

B = np.array([[-1,0],[0,1]])
B_coords = B@coords

x_LT2 = B_coords[0,:]
y_LT2 = B_coords[1,:]

fig, ax = plt.subplots()

ax.plot(x,y,'ro')
ax.plot(x_LT2,y_LT2,'bo')

ax.plot(x,y,'r',ls="--")
ax.plot(x_LT2,y_LT2,'b')


ax.axvline(x=0,color="k",ls=":")
ax.axhline(y=0,color="k",ls=":")
ax.grid(True)
ax.axis([-4,4,-1,2])
ax.set_aspect('equal')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xticks(np.arange(-3.5,4,step = 0.5))
ax.set_yticks(np.arange(-1,2,step = 0.5))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.grid(True,ls=':')