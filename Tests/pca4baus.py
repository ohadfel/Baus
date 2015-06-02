import itertools
import pickle

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
import time
from sklearn import neighbors
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from scipy.sparse import csr_matrix



if socket.gethostname() == 'Ohad-PC':
    base_path = 'C:\Users\Ohad\Copy\Baus'

elif socket.gethostname()[:3] == 'ctx':
    base_path = '/home/lab/ohadfel/Documents/Baus'
else:
    base_path = '/media/ohadfel/New Volume/Copy/Baus'
    #  base_path = '/home/ohadfel/Copy/Baus'

print(base_path)

DUMP_FOLDER = os.path.join(base_path, 'dumps')
CURRENT_DUMP_FOLDER = ''
# DUMP_FOLDER = '/home/ohadfel/Copy/Baus/dumps'
# DUMP_FOLDER = 'C:\Users\Ohad\Copy\Baus\dumps'

# DUMP_FOLDER = '/media/ohadfel/New\ Volume/Results'
SAVE = False
NUM_OF_CLUSTERS = 2
NUM_OF_NEIGHBORS = 3

n_clusters = [5, 25, 50, 100, 250, 500, 1000]
n_neighbors = [3, 7, 15, 25, 33, 67, 133, 277, 555, 1111]
tuned_parameters = []
for i, j in itertools.product(n_clusters, n_neighbors):
    if i*2 > j:
        tuned_parameters.append((i, j))


def run(x, y, x_test, y_test, folds_num, path, inds, jobs_num=6, calc_probs=True):
    # cv = StratifiedShuffleSplit(y, folds_num, heldoutSize, random_state=0)
    print('dataset is :'+path)
    # scaler = preprocessing.StandardScaler().fit(x)
    # x = scaler.transform(x)

    # fig = plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    # plt.cla()
    # pca = decomposition.PCA(n_components=3)

    pca = decomposition.TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
    # pca = decomposition. RandomizedPCA(n_components=2,copy=False)
    pca.fit(x)
    X = pca.transform(x)

    # for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    #     ax.text3D(X[y == label, 0].mean(),
    #               X[y == label, 1].mean() + 1.5,
    #               X[y == label, 2].mean(), name,
    #               horizontalalignment='center',
    #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(np.float)

    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    # plt.set_xlabel('PC1')
    # plt.set_ylabel('PC2')
    # ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.seismic)
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    plt.title(path + ' PCA')
    plt.show()


def calc_cv_scores(p):
    x, y, train_index, test_index, tuned_param, fold_num, centroids, labels = p
    scaler = preprocessing.StandardScaler().fit(x[train_index])
    x_test = scaler.transform(x[test_index])

    t = time.time()

    print('Training on KNN...')
    clf = neighbors.KNeighborsClassifier(tuned_param[1])
    clf.fit(centroids, labels)

    print('Predicting with KNN...')
    ypred = clf.predict(x_test)
    ytrue = y[test_index]
    score = calculate_score(ypred, ytrue)

    elapsed = time.time() - t
    print(tuned_param)
    print('Request took '+str(elapsed)+' sec.')
    return clf, score


def calculate_centroids(p):
    x, y, train_index, test_index, tuned_param, fold_num = p
    scaler = preprocessing.StandardScaler().fit(x[train_index])
    mini_x0, mini_y0, mini_x1, mini_y1 = split_between_labels(x, y, train_index)

    mini_x0 = scaler.transform(mini_x0)
    mini_x1 = scaler.transform(mini_x1)
    x_test = scaler.transform(x[test_index])

    t = time.time()
    print('Clustering...')
    kmeans = cluster.KMeans(n_clusters=tuned_param[0])
    kmeans.fit(mini_x0)
    # labels0 = kmeans.labels_
    centroids0 = kmeans.cluster_centers_
    labels = np.zeros(len(centroids0))

    kmeans = cluster.KMeans(n_clusters=tuned_param[0])
    kmeans.fit(mini_x1)
    # labels1 = kmeans.labels_
    centroids1 = kmeans.cluster_centers_
    tmp = np.ones(len(centroids1))
    labels = np.concatenate((labels, tmp), axis=0)
    centroids = np.concatenate((centroids0, centroids1), axis=0)
    return centroids, labels


def split_between_labels(x, y, train_index):
    mini_x = x[train_index, :]
    mini_y = y[train_index]
    mini_x0 = mini_x[mini_y == 0, :]
    mini_y0 = mini_y[mini_y == 0]
    mini_x1 = mini_x[mini_y == 1, :]
    mini_y1 = mini_y[mini_y == 1]
    return mini_x0, mini_y0, mini_x1, mini_y1


def calculate_score(ypred, ytrue):
    correct_ans = sum(ypred == ytrue)
    total_num = float(len(ypred))
    print(correct_ans/total_num)
    return correct_ans/total_num


def report(params, path, score, cv_scores):
    log_str = os.path.join(DUMP_FOLDER, 'logKNN.txt')
    f = open(log_str, 'a')
    f.seek(0)  # get to the first position
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCORE='+str(score)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    line = 'params '+str(params)+' dataset '+path+' score '+str(score)+'CV scores=' + str(cv_scores) + "\n"
    f.write(line)
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    f.close()

    sum_str = os.path.join(DUMP_FOLDER, 'summaryKNN.txt')
    f = open(sum_str, 'a')
    f.seek(0)  # get to the first position
    line = 'number of clusters=;'+str(params[0])+';number of neighbors=;'+str(params[1])+';dataset '+path+';score;'+str(sum(cv_scores)/len(cv_scores)) + "\n"
    f.write(line)
    f.close()


def check_if_params_were_calculated(dump_path, tuned_param):
    return False


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


def load_pickled_data(path):
    file_path = os.path.join(path, 'Xtrain.pkl')
    data = pickle.load(open(file_path, 'rb'))
    x = data.T

    file_path = os.path.join(path, 'Ytrain.pkl')
    data = pickle.load(open(file_path, 'rb'))
    y = np.squeeze(data)
    y = y.T

    file_path = os.path.join(path, 'Xtest.pkl')
    data = pickle.load(open(file_path, 'rb'))
    x_test = data  # For converting to numpy array

    x_test = x_test.T

    file_path = os.path.join(path, 'Ytest.pkl')
    data = pickle.load(open(file_path, 'rb'))
    y_test = data  # For converting to numpy array
    y_test = np.squeeze(y_test)
    y_test = y_test.T

    file_path = os.path.join(path, 'inds.pkl')
    data = pickle.load(open(file_path, 'rb'))
    inds = np.array(data, dtype=np.int)

    return x, y, x_test, y_test, inds


def load_sparse_data(path):
    os.path.join(path, 'Xtrain.mat')
    f = h5py.File(os.path.join(path, 'Xtrain.mat'), 'r')
    data = f.get('XTrain')
    x = np.array(data)  # For converting to numpy array
    x = x.T

    x = csr_matrix((x[:,2], (x[:,0], x[:,1])), shape=(27438, 149331))
    # x = x.toarray()

    f = h5py.File(os.path.join(path, 'Ytrain.mat'), 'r')
    data = f.get('YTrain')
    y = np.array(data)  # For converting to numpy array
    y = np.squeeze(y)
    y = y.T

    f = h5py.File(os.path.join(path, 'Xtest.mat'), 'r')
    data = f.get('XTest')
    x_test = np.array(data)  # For converting to numpy array
    x_test = x_test.T

    x_test = csr_matrix((x_test[:, 2], (x_test[:, 0], x_test[:, 1])), shape=(8429, 149331))
    # x_test = x_test.toarray()

    f = h5py.File(os.path.join(path, 'Ytest.mat'), 'r')
    data = f.get('YTest')
    y_test = np.array(data)  # For converting to numpy array
    y_test = np.squeeze(y_test)
    y_test = y_test.T

    f = h5py.File(os.path.join(path, 'inds.mat'), 'r')
    inds = f.get('patternsCrossValInd')
    inds = np.array(inds, dtype=np.int)

    return x, y, x_test, y_test, inds


def load_pickled_sparse_data(path):
    file_path = os.path.join(path, 'Xtrain.pkl')
    data = pickle.load(open(file_path, 'rb'))
    x = data.T

    file_path = os.path.join(path, 'Ytrain.pkl')
    data = pickle.load(open(file_path, 'rb'))
    y = np.squeeze(data)
    y = y.T

    file_path = os.path.join(path, 'Xtest.pkl')
    data = pickle.load(open(file_path, 'rb'))
    x_test =data  # For converting to numpy array

    x_test = x_test.T

    file_path = os.path.join(path, 'Ytest.pkl')
    data = pickle.load(open(file_path, 'rb'))
    y_test = data  # For converting to numpy array
    y_test = np.squeeze(y_test)
    y_test = y_test.T

    file_path = os.path.join(path, 'inds.pkl')
    data = pickle.load(open(file_path, 'rb'))
    inds = np.array(data, dtype=np.int)

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
    # f = h5py.File(os.path.join(path, 'Ytrain.mat')
    # inds = f.get('patternsCrossValInd')
    # inds = np.array(inds, dtype=np.int)
    # path='/home/ohadfel/Desktop/4ohad/Last_change'
    # path='/home/ohadfel/Copy/Baus/Pre/data1'
    for feature_folder in differentDataPaths:
        # path = os.path.join(base_path, 'Pre', 'data1')
        if feature_folder.find('sparse') != -1:
            if socket.gethostname()[:3] == 'ctx':
                x, y, x_test, y_test, inds = load_pickled_sparse_data(feature_folder)
            else:
                x, y, x_test, y_test, inds = load_sparse_data(feature_folder)
        else:
            if True or socket.gethostname()[:3] == 'ctx':
                x, y, x_test, y_test, inds = load_pickled_data(feature_folder)
            else:
                x, y, x_test, y_test, inds = load_data(feature_folder)
        print(feature_folder)
        folds_num = 5

        scaler = preprocessing.StandardScaler(with_mean=False).fit(x)
        x = scaler.transform(x)
        run(x, y, x_test, y_test, folds_num, feature_folder, inds, 5)
    print('finish!')


if __name__ == '__main__':
    cProfile.run('print go(); print')