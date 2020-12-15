# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:29:54 2020

@author: Ben Vanderlei
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

options = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}
ax.arrow(0,0,1,3,fc='b',ec='b',**options)
# ax.arrow(3,0,1,3,fc='b',ec='b',**options)
# ax.arrow(0,2,1,3,fc='b',ec='b',**options)
# ax.arrow(2,1,1,3,fc='b',ec='b',**options)

ax.text(1,2,'$U_1$')

ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.set_aspect('equal')
ax.grid(True,ls=':')
