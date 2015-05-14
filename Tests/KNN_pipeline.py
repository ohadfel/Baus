__author__ = 'ohadfel'
# create centroids for each condition
# when need to predict do majority
# check the results
# if results are nice you can represent the data using distances from all centroids

import cProfile
import os
import socket
import h5py
from Tests import utils
from crosVal import IndicesKFold
from sklearn import cluster
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors


if socket.gethostname() == 'Ohad-PC':
    base_path = 'C:\Users\Ohad\Copy\Baus'
else:
    base_path = '/media/ohadfel/New Volume/Copy/Baus'
    #  base_path = '/home/ohadfel/Copy/Baus'

DUMP_FOLDER = os.path.join(base_path, 'dumps')
CURRENT_DUMP_FOLDER = ''
# DUMP_FOLDER = '/home/ohadfel/Copy/Baus/dumps'
# DUMP_FOLDER = 'C:\Users\Ohad\Copy\Baus\dumps'

# DUMP_FOLDER = '/media/ohadfel/New\ Volume/Results'
SAVE = False
k = 2

tuned_parameters = []


def run(x, y, x_test, y_test, folds_num, path, inds, jobs_num=6, calc_probs=True):
    # cv = StratifiedShuffleSplit(y, folds_num, heldoutSize, random_state=0)
    cv = IndicesKFold(inds, folds_num)  # , 4000, 1000)
    params = []
    for fold_num, (train_index, test_index) in enumerate(cv):
        params.append((x, y, train_index, test_index, tuned_param, fold_num))
        print(tuned_param)
        if jobs_num == 1:
            map_results = [calc_cv_scores(p) for p in params]  # For debugging
        else:
            map_results = utils.parmap(calc_cv_scores, params, jobs_num)
        print('so far so good.')
        mini_x = x[train_index, :]
        mini_y = y[train_index]
        mini_x0 = mini_x[mini_y == 0, :]
        mini_y0 = mini_y[mini_y == 0]
        mini_x1 = mini_x[mini_y == 1, :]
        mini_y1 = mini_y[mini_y == 1]

        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(mini_x0)
        # labels0 = kmeans.labels_
        centroids0 = kmeans.cluster_centers_

        kmeans.fit(mini_x1)
        # labels1 = kmeans.labels_
        centroids1 = kmeans.cluster_centers_




def calc_cv_scores(p):
    x, y, train_index, test_index, tuned_param, fold_num = p
    scaler = preprocessing.StandardScaler().fit(x[train_index])
    mini_x0, mini_y0, mini_x1, mini_y1 = split_between_labels(x, y, train_index)

    mini_x0 = scaler.transform(mini_x0)
    mini_x1 = scaler.transform(mini_x1)
    x_test = scaler.transform(x[test_index])

    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(mini_x0)
    # labels0 = kmeans.labels_
    centroids0 = kmeans.cluster_centers_
    labels = np.zeros(len(centroids0))

    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(mini_x1)
    # labels1 = kmeans.labels_
    centroids1 = kmeans.cluster_centers_
    tmp = np.ones(len(centroids1))
    labels = np.concatenate((labels, tmp), axis=0)
    centroids = np.concatenate((centroids0, centroids1), axis=0)






    clf = TSVC(C=tuned_param['C'][0], kernel=tuned_param['kernel'][0], gamma=tuned_param.get('gamma', [0])[0], calc_probs=calc_probs)
    print(fold_num, str(datetime.now()))
    print('number of train samples'+str(x[train_index].shape), 'number of test samples'+str(x[test_index].shape))
    print(Counter(y))
    t = time.time()
    clf.fit(x[train_index], y[train_index])
    ypred = clf.predict(x[test_index])
    score = auc_score(ypred, y[test_index])
    elapsed = time.time() - t
    print(tuned_param)
    print('Request took '+str(elapsed)+' sec.')
    return clf, score



def split_between_labels(x, y, train_index):
    mini_x = x[train_index, :]
    mini_y = y[train_index]
    mini_x0 = mini_x[mini_y == 0, :]
    mini_y0 = mini_y[mini_y == 0]
    mini_x1 = mini_x[mini_y == 1, :]
    mini_y1 = mini_y[mini_y == 1]
    return mini_x0, mini_y0, mini_x1, mini_y1



def load_data(path):
    os.path.join(path, 'Xtrain.mat')
    f = h5py.File(os.path.join(path, 'Xtrain.mat'), 'r')
    data = f.get('XTrain')
    x = np.array(data)  # For converting to numpy array
    x = x.T

    f = h5py.File(os.path.join(path, 'Ytrain.mat'), 'r')
    data = f.get('YTrain')
    y = np.array(data)  # For converting to numpy array
    y = np.squeeze(y)
    y = y.T

    f = h5py.File(os.path.join(path, 'Xtest.mat'), 'r')
    data = f.get('XTest')
    x_test = np.array(data)  # For converting to numpy array
    x_test = x_test.T

    f = h5py.File(os.path.join(path, 'Ytest.mat'), 'r')
    data = f.get('YTest')
    y_test = np.array(data)  # For converting to numpy array
    y_test = np.squeeze(y_test)
    y_test = y_test.T

    f = h5py.File(os.path.join(path, 'inds.mat'), 'r')
    inds = f.get('patternsCrossValInd')
    inds = np.array(inds, dtype=np.int)

    return x, y, x_test, y_test, inds


def go():
    directory = os.path.join(base_path, 'Pre')
    differentDataPaths = [x[0] for x in os.walk(directory)]
    differentDataPaths = differentDataPaths[1:]
    # if socket.gethostname() == 'Ohad-PC':
    #     base_path = 'C:\Users\Ohad\Copy\Baus'
    # else:
    #     base_path = '/home/ohadfel/Copy/Baus'

    # path = os.path.join(basePath, 'Code', 'matlab', 'inds.mat')
    # path = os.path.join(base_path, 'Pre', 'data1', 'inds.mat')

    # f = h5py.File(path, 'r')
    # inds = f.get('patternsCrossValInd')
    # inds = np.array(inds, dtype=np.int)
    # path='/home/ohadfel/Desktop/4ohad/Last_change'
    # path='/home/ohadfel/Copy/Baus/Pre/data1'
    for feature_folder in differentDataPaths:
        # path = os.path.join(base_path, 'Pre', 'data1')
        x, y, x_test, y_test, inds = load_data(feature_folder)
        print(feature_folder)
        folds_num = 5

        run(x, y, x_test, y_test, folds_num, feature_folder, inds, 1)
    print('finish!')


if __name__ == '__main__':
    cProfile.run('print go(); print')