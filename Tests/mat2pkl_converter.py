__author__ = 'ohadfel'
# 19/05/2015
import h5py
import pickle
import os
import numpy as np

base_path = '/media/ohadfel/New Volume/Copy/Baus'

def save(obj, file_name):
    with open(file_name, 'w') as pklFile:
        pickle.dump(obj, pklFile)


def convert(file):
    f = h5py.File(file, 'r')
    for cur_var in f:
        if cur_var in ['XTrain', 'YTrain', 'XTest', 'YTest', 'patternsCrossValInd']:
            var = np.array(f.get(cur_var))

    new_file_parts = file.split('.')
    new_file_parts[-1] = 'pkl'
    new_file = '.'.join(new_file_parts)
    save(var, new_file)


def find_var(name):
    if name == 'inds.mat':
        return 'patternsCrossValInd'
    else:
        new_name = name[0]+'T'+name[2:]
        return new_name

if __name__ == '__main__':
    directory = os.path.join(base_path, 'Pre')
    differentDataPaths = [x[0] for x in os.walk(directory)]
    differentDataPaths = differentDataPaths[1:]
    for feature_folder in differentDataPaths:
        for files in os.walk(os.path.join(directory, feature_folder)):
            for file in files[2]:
                if file.endswith(".mat"):
                    convert(os.path.join(feature_folder, file))