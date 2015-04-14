'''
Created on Mar 30, 2015

@author: ohadfel
'''
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transition=np.linspace(0,1,10)
    zero=np.linspace(0,0,100)
    one=np.linspace(1,1,30)
    
    full=np.concatenate((zero,transition,one,transition[::-1],zero), axis=0)
    x=np.arange(0,len(full))
    line, = plt.plot(x, full,linewidth=2)
    plt.show()