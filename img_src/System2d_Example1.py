# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:25:52 2020

@author: Ben Vanderlei
"""

import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-5,5,100)

fig, ax = plt.subplots()


ax.plot(x,(5-x)/3)
ax.plot(x,(5+x)/2)

ax.text(1,1.6,'$x_1+3x_2 = 5$')
ax.text(-3,0.5,'$x_1-2x_2 = -5$')

ax.set_xlim(-4,4)
ax.set_ylim(-2,6)
ax.axvline(color='k',linewidth = 1)
ax.axhline(color='k',linewidth = 1)

## Ticks!
xticks = []
for i in range(-4,5):
    xticks.append(i)
ax.set_xticks(xticks)

ax.set_aspect('equal')
ax.grid(True,ls=':')

#fig.savefig('System2d_Example1.png')