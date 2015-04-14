'''
Created on Mar 13, 2015

@author: ohadfel
'''

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib import rcParams
from matplotlib import animation
from matplotlib.figure import Figure
import matplotlib.cm as cmx
from mpldatacursor import datacursor
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.base import Bunch

import pylab
import scipy.cluster.hierarchy as sch

def scatter3d(X, fig=None,ax=None ,color='b',cs=None, colorsMap='jet'):
    if (cs is not None):
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    if (ax is None):
        fig = plt.figure()
        ax = Axes3D(fig)
    if (cs is None):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=color)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    b=Bunch()
    b.fig=fig
    b.ax=ax
    return b
    
def findBrainAxis(X):
    mapping=[0,1,2]
    Diffs=[(max(X[:, ii])-min(X[:, ii])) for ii in range(3)]
    mapping[0]=Diffs.index(max(Diffs))
    mins=[(min(X[:, ii])) for ii in range(3)]
    mapping[2]=mins.index(max(mins))
    mapping[1]=3-(mapping[0]+mapping[2])
    return mapping
    
            