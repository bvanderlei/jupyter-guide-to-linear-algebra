# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:54:59 2021

@author: Ben Vanderlei
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()

coords_Rn = np.array([[0,0],[0,3],[4,5],[4,2],[0,0]])
coords_Rm = np.array([[8,1],[8,5],[13,7],[13,3],[8,1]])

x_Rn = coords_Rn[:,0]
y_Rn = coords_Rn[:,1]
x_Rm = coords_Rm[:,0]
y_Rm = coords_Rm[:,1]

ax.plot(x_Rn,y_Rn)
ax.plot(x_Rm,y_Rm,'g')

options = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}
ax.arrow(0,0,2,3,fc='b',ec='b',**options)
ax.arrow(8,1,3,3,fc='g',ec='g',**options)

options = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", 
               color = 'k')
T = patches.FancyArrowPatch((1,1),(8.5,2.5),connectionstyle='arc3,rad=-0.25',
                            **options)
ax.add_patch(T)

ax.set_xlim(-2,15)
ax.set_ylim(0,7)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.text(1,4.1,'$\mathbb{R}^n$',fontsize=18);
ax.text(0.8,2,'$V$',fontsize=16);
ax.text(5,3.25,"T",fontsize=16);
ax.text(9,6.0,'$\mathbb{R}^m$',fontsize=18);
ax.text(8.4,3.5,'$W = T(V)$',fontsize=16);
fig.set_figheight(8)
fig.set_figwidth(12)