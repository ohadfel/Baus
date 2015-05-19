__author__ = 'ohadfel'
import numpy as np

if __name__ == '__main__':
    fname="/home/lab/ohadfel/Documents/Baus/dumps/summary.txt"
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
    print('Finish!!')





