# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:19:52 2021

Visualization of span

@author: Ben Vanderlei
"""
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,6)
ax.grid(False)

xx, yy = np.meshgrid(range(6), range(6))
z = (4*xx + 6*yy)/9 

ax.plot_surface(xx, yy, z, alpha=0.5)

options = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", 
               color = 'k')

ax.arrow3D(0,0,0,0,3,2,**options)
ax.arrow3D(0,0,0,3,4,4,**options)
ax.arrow3D(0,0,0,3,1,2,**options)

ax.text(0.2,5,8/3,"Span",(3,2,0),size="x-large")
ax.text2D(0.1,0.2,"The span of a set of vectors in $\mathbb{R}^3$ may form a plane.",
          transform=ax.transAxes)

# No ticks
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# Make panes transparent
# ax.xaxis.pane.fill = False # Left pane
# ax.yaxis.pane.fill = False # Right pane

# Transparent spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Transparent panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

fig.set_figheight(8)
fig.set_figwidth(12)
