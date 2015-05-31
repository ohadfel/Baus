__author__ = 'ohadfel'
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fname = "/media/ohadfel/New Volume/Copy/Baus/dumps/summary.txt"
    with open(fname) as f:
        content = f.readlines()

    c = []
    gamma = []
    scores = []
    datasets = []
    for l in content:
        l_by_commas = l.split(',')
        c.append(float(l_by_commas[0].split('=')[-1]))
        gamma.append(float(l_by_commas[2].split('=')[-1]))
        l_by_semicolon = l.split(';')
        scores.append(float(l_by_semicolon[-1][:-1]))
        datasets.append(l_by_semicolon[-3])
    c_values = sorted(list(set(c)))
    gamma_values = sorted(list(set(gamma)))
    datasets_values = sorted(list(set(datasets)))
    full_matrix = np.empty((len(datasets_values), len(c_values), len(gamma_values)))
    full_matrix[:] = np.NAN

    for ii in range(0, len(c)):
        full_matrix[datasets_values.index(datasets[ii]), c_values.index(c[ii]), gamma_values.index(gamma[ii])] = scores[ii]
    # np.savetxt('full_matrix_saving_test.txt', full_matrix, delimiter=",", fmt="%s")

    alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
    matrix,rows,cols = full_matrix.shape
    for dataset in range(matrix):
        plt.subplot(matrix, 3, dataset*3)
        for i in range(rows):
            axhline(y=i,color='k')
        for j in range(cols):
            axvline(x=j,color='k')
        for i in range(rows):
            for j in range(cols):
                theString=str(full_matrix[dataset,i,j])
                iOffset=1/float(rows)*0.5
                jOffset=1/float(cols)*0.5

                t=text(j+0.5,i+0.5, theString[:6], **alignment)
        xlim(0, 9)
        ylim(3, 0)
        xticks(arange(9)+0.5, ('1e-6','1e-5','1e-4','1e-3','1e-2','1e-1','1e0','1e1','1e2'))
        yticks(arange(3)+0.5, c_values)
        ylabel('C')
        xlabel('gamma')
        title(datasets_values[dataset].split('-')[1])
    print('Finish!!')





