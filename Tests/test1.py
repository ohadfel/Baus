'''
Created on Mar 12, 2015

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


def matLoad(matfile):
    mat = scipy.io.loadmat(matfile)
    matStruct=BunchDic(mat)
    return matStruct


if __name__ == '__main__':
    #import mat files from Matlab
    mat = matLoad('/home/ohadfel/Desktop/4ohad/move2py.mat')
    pos=mat.newBalls
    vis.findBrainAxis(pos)
    nansArr=np.array([float('nan'),float('nan'),float('nan')])
    ballsWithoutSeedAndSec=pos.copy()
#     seed=402
#     second=316
    seed=33
    second=203
    ballsWithoutSeedAndSec[seed]=nansArr
    ballsWithoutSeedAndSec[second]=nansArr
    fig = plt.figure()
#     ax = Axes3D(fig)
    ax=fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(ballsWithoutSeedAndSec[:, 0], ballsWithoutSeedAndSec[:, 1], ballsWithoutSeedAndSec[:, 2],c='b',s=20)
    ax.scatter(pos[seed, 0], pos[seed, 1], pos[seed, 2],c='r',s=40)
    ax.scatter(pos[second, 0], pos[second, 1], pos[second, 2],c='c',s=40)

    dist = np.linalg.norm(np.array([pos[33, 0], pos[33, 1], pos[33, 2]])-np.array([pos[203, 0], pos[203, 1], pos[203, 2]]))
    ax.set_title('Cross correlation '+str(seed)+'X'+str(second)+'(distance='+str(dist)+')')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    
#     print('Calculate cross correlation')
# #     cross1=np.correlate(mat.CCD547[33], mat.CCD547[203])
#     cross1=np.correlate(mat.congruentTrialsCat[33,:],mat.congruentTrialsCat[191,:])
#     ax1 = fig.add_subplot(2,2,2)
#     tmp=mat.CCD547[33]
#     print(tmp.shape)
# #     ax1.xcorr(mat.CCD547[33],mat.CCD547[203], usevlines=True, maxlags=20,normed=True, lw=2)
# #     ax1.grid(True)
# #     ax1.axhline(0, color='black', lw=2)
#     ax1.plot(range(len(cross1)),cross1)
# #     plt.show()
    
    print('second scatter')
    ballsWithoutSeedAndSec=pos.copy()
    ballsWithoutSeedAndSec[33]=nansArr
    ballsWithoutSeedAndSec[190]=nansArr
#     fig = plt.figure()
#     ax = Axes3D(fig)
    ax=fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(ballsWithoutSeedAndSec[:, 0], ballsWithoutSeedAndSec[:, 1], ballsWithoutSeedAndSec[:, 2],c='b',s=20)
    ax.scatter(pos[33, 0], pos[33, 1], pos[33, 2],c='r',s=40)
    ax.scatter(pos[190, 0], pos[190, 1], pos[190, 2],c='g',s=40)
    dist = np.linalg.norm(np.array([pos[33, 0], pos[33, 1], pos[33, 2]])-np.array([pos[190, 0], pos[190, 1], pos[190, 2]]))
    ax.set_title('Cross correlation 33X190 (distance='+str(dist)+')')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    plt.show()
    
    
    
    
    
#     b=vis.scatter3d(pos)
#     vis.scatter3d(pos[100:102,:],b.fig,b.ax,'r')
#     df=pd.DataFrame(mat.CCD547)  
      
    print('Hello World')
