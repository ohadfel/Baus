__author__ = 'ohadfel'
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc
from scipy import interp
import h5py
import pickle
import os
import matplotlib.pyplot as plt
import cProfile

import numpy as np
import sklearn
from datetime import datetime
import time
from crosVal import IndicesKFold
import utils
from collections import Counter
import socket
from scipy.sparse import csr_matrix


# np.logspace(-2, 3, 6)

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

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1], 'C': [1]}, {'kernel': ['rbf'], 'gamma': [1e-2], 'C': [1]},
                    {'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1]}, {'kernel': ['rbf'], 'gamma': [1e-4], 'C': [1]},
                    {'kernel': ['rbf'], 'gamma': [1e-5], 'C': [1]}, {'kernel': ['rbf'], 'gamma': [1e-6], 'C': [1]},
                    {'kernel': ['rbf'], 'gamma': [1e-1], 'C': [0.1]}, {'kernel': ['rbf'], 'gamma': [1e-2], 'C': [0.1]},
                    {'kernel': ['rbf'], 'gamma': [1e-3], 'C': [0.1]}, {'kernel': ['rbf'], 'gamma': [1e-4], 'C': [0.1]},
                    {'kernel': ['rbf'], 'gamma': [1e-5], 'C': [0.1]}, {'kernel': ['rbf'], 'gamma': [1e-6], 'C': [0.1]},
                    {'kernel': ['rbf'], 'gamma': [1e-1], 'C': [0.01]}, {'kernel': ['rbf'], 'gamma': [1e-2], 'C': [0.01]},
                    {'kernel': ['rbf'], 'gamma': [1e-3], 'C': [0.01]}, {'kernel': ['rbf'], 'gamma': [1e-4], 'C': [0.01]},
                    {'kernel': ['rbf'], 'gamma': [1], 'C': [1]}, {'kernel': ['rbf'], 'gamma': [10], 'C': [1]},
                    {'kernel': ['rbf'], 'gamma': [1e2], 'C': [1]}, {'kernel': ['rbf'], 'gamma': [1], 'C': [0.1]},
                    {'kernel': ['rbf'], 'gamma': [10], 'C': [0.1]}]  # , {'kernel': ['rbf'], 'gamma': [1e-3], 'C': [10]},
                    # {'kernel': ['rbf'], 'gamma': [1e-2], 'C': [10]}, {'kernel': ['rbf'], 'gamma': [1e-1], 'C': [10]},
                    # {'kernel': ['rbf'], 'gamma': [1], 'C': [10]}, {'kernel': ['rbf'], 'gamma': [10], 'C': [10]}]

def run(x, y, x_test, y_test, folds_num, path, inds, jobs_num=6, calc_probs=True):
    # y = y.astype(np.int)
    # y_test = y_test.astype(np.int)

    # heldoutSize = 1./folds_num
    # x, idx = shuffle(x)
    # y = y[idx]
    # x_test, idx = shuffle(x_test)
    # y_test = y_test[idx]

    # scaler = preprocessing.StandardScaler().fit(x)
    # x = scaler.transform(x)
    # x_test = scaler.transform(x_test)

    # x = x[:5000, :]
    # y = y[:5000]
    # ---------------------------------------------------- x_test = x_test[:500, :]
    # ------------------------------------------------------- y_test = y_test[:500]

    # cv = StratifiedShuffleSplit(y, folds_num, heldoutSize, random_state=0)
    cv = IndicesKFold(inds, folds_num)  # , 4000, 1000)
    # -------------------------------- scores = ['roc_auc','precision', 'recall']
    # -------------- auc_scoreFunc = make_scorer(auc_score, greater_is_better=True)
    # scores = ['roc_auc'] #[auc_scoreFunc] # 'roc_auc'
    #  ---------------------------------------------------- scores = ['precision']
    # calc_probs = True

    parts_of_path = path.split('/')
    dump_path = os.path.join(DUMP_FOLDER, parts_of_path[-1])
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    current_dump_folder = dump_path
    print('Start the grid search')
    t = time.time()
    for tuned_param in tuned_parameters:
        already_exist = check_if_params_were_calculated(dump_path, tuned_param)
        if already_exist:
            continue
        params = []
        for fold_num, (train_index, test_index) in enumerate(cv):
            params.append((x, y, train_index, test_index, tuned_param, fold_num, calc_probs))

        print(tuned_param)
        if jobs_num == 1:
            map_results = [calc_cv_scores(p) for p in params]  # For debugging
        else:
            map_results = utils.parmap(calc_cv_scores, params, jobs_num)

        cv_scores = np.array([score for (clf, score) in map_results][:len(cv)])
        print(cv_scores)
        mean_cv_score = sum(cv_scores)/len(cv_scores)
        print('==============================mean auc score is '+str(mean_cv_score)+'==============================')
        clf = map_results[0][0]

        elapsed = time.time() - t
        print('Request took '+str(elapsed)+' sec.')
        print(str(datetime.now()))

        # scaler = preprocessing.StandardScaler().fit(x)
        # x = scaler.transform(x)
        # x_test = scaler.transform(x_test)

        mini_path = path.split('/')[-1]
        mini_path = mini_path.replace(' ', '_')
        print_results(clf, x_test, y_test, calc_probs, path, None, cv_scores, current_dump_folder, mean_cv_score)

class TSVC(SVC):
    def __init__(self, C=1, kernel='rbf', gamma=0, calc_probs=True):
        super(TSVC, self).__init__(C=C, kernel=kernel, gamma=gamma, probability=True)
        self.calc_probs = calc_probs
        # ----------------------------------------------- print("in constructor")
        # print('C='+str(C)+', kernel='+kernel+', gamma='+str(gamma)+', probability=True')

    def fit(self, x, y, do_shuffle=True):
        # if (do_shuffle):
        #     (x, idx) = shuffle(x)
        # y = y[idx]
        # self.scaler = preprocessing.StandardScaler().fit(x)
        # x = self.scaler.transform(x)
        super(TSVC, self).fit(x, y)
        return self

    def predict(self, x):
        print('predict!')
        #         x = self.scaler.transform(x)
        if self.calc_probs:
            probs = super(TSVC, self).predict_proba(x)
        else:
            probs = super(TSVC, self).predict(x)
        # score = roc_auc_score(self.ytrue, probs[:,1])
        return probs

def save(obj, file_name):
    if SAVE:
        full_file_name = os.path.join(CURRENT_DUMP_FOLDER, file_name)
        with open(full_file_name, 'w') as pklFile:
            pickle.dump(obj, pklFile)

def probs_to_preds(probs):
    return np.array([0 if p[0] > 0.5 else 1 for p in probs])

def report(params, path, score, dump_file_name, time, cv_scores, cm_normalized):
    # AUC Score=88,
    log_str = os.path.join(DUMP_FOLDER, 'log.txt')
    f = open(log_str, 'a')
    f.seek(0)  # get to the first position
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCORE='+str(score)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    line = 'params '+params+' dataset '+path+' score '+str(score)+' dump File Name '+dump_file_name+' Time was '+str(time) + "\n"
    f.write(line)
    line = 'CV scores=' + str(cv_scores) + ' CM='+str(cm_normalized) + "\n"
    f.write(line)
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    f.close()

    sum_str = os.path.join(DUMP_FOLDER, 'summary.txt')
    f = open(sum_str, 'a')
    f.seek(0)  # get to the first position
    line = 'params '+params+';dataset '+path+';score;'+str(sum(cv_scores)/len(cv_scores)) + "\n"
    f.write(line)
    f.close()

def calc_auc(probs, y, do_plot=True, roc_fig_name=''):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    if roc_fig_name != '':
        do_plot = False
    return plot_roc(mean_tpr, mean_fpr, do_plot, file_name=roc_fig_name)

def plot_roc(mean_tpr, mean_fpr, do_plot, len_cross_validation=1, file_name=''):
    if do_plot:
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

    mean_tpr /= float(len_cross_validation)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # if (do_plot):
    plt.ion()
    plt.plot(mean_fpr, mean_tpr, 'r-', label='Mean ROC (area = %0.2f)' % mean_auc, lw=3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if file_name != '' and SAVE:
        plt.savefig(file_name)

    if do_plot:
        plt.show()
    else:
        plt.close()
    return mean_auc, mean_tpr, mean_fpr

def load_sparse_data(path):
    os.path.join(path, 'Xtrain.mat')
    f = h5py.File(os.path.join(path, 'Xtrain.mat'), 'r')
    data = f.get('XTrain')
    x = np.array(data)  # For converting to numpy array
    x = x.T

    x = csr_matrix((x[:,2], (x[:,0], x[:,1])), shape=(27438, 149331))
    x = x.toarray()

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
    x_test = x_test.toarray()

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

        run(x, y, x_test, y_test, folds_num, feature_folder, inds, 5)
    print('finish!')

if __name__ == '__main__':
    cProfile.run('print go(); print')