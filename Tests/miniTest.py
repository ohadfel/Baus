'''
Created on Mar 15, 2015

@author: ohadfel
'''
import scipy.io
from utilities.utils import BunchDic
import pandas as pd
import visualization.vis as vis
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy


def matLoad(matfile):
    mat = scipy.io.loadmat(matfile)
    matStruct=BunchDic(mat)
    return matStruct


if __name__ == '__main__':
    mat = matLoad('/home/ohadfel/Desktop/4ohad/move2py.mat')
    x,y = mat.congruentTrialsCat[33],mat.congruentTrialsCat[203]
#     xCorr=np.correlate(x, y, mode='full')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    Xcorr=np.correlate(x, y, mode=2)
    norm=np.sqrt(np.dot(x, x) * np.dot(y, y))
#     norm=1.5
    print(norm)
    Xcorr /= norm
#     X= np.arange(-maxlags, maxlags + 1)
    Nx = len(x)
    lags = np.arange(-550, 550 + 1)
    Xcorr = Xcorr[Nx - 1 - 550:Nx + 550]
    ax1.plot(lags,Xcorr)
#     ax1.plot(xCorr)
#     ax1.xcorr(x, y, usevlines=True, maxlags=100, normed=False, lw=1)
    ax1.grid(True)
    ax1.axhline(0, color='black', lw=2)
    
    ax2 = fig.add_subplot(212)
    ax2.acorr(x, usevlines=True, normed=True, maxlags=600, lw=2)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=2)
    
    plt.show()
