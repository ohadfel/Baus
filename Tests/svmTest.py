#
# Created on Mar 23, 2015
#
# @author: ohadfel
#

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

# class Timer(object):
#     def __init__(self, name=None, doPrint = True):
#         self.name = name
#         self.doPrint = doPrintos.path.join(basePath, 'Pre', 'data1')
#
#     def __enter__(self):
#         self.tstart = time.time()
#
#     def __exit__(self, type, value, traceback):
#         if self.doPrint:
#             if self.name:
#                 print '[%s]' % self.name,
#             calc=(time.time() - self.tstart)
#             print 'Elapsed: '+str(calc)
#         return calc


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
        print_results(clf, x_test, y_test, calc_probs, path, None, cv_scores, current_dump_folder,mean_cv_score)


def calc_cv_scores(p):
    x, y, train_index, test_index, tuned_param, fold_num, calc_probs = p

    scaler = preprocessing.StandardScaler().fit(x[train_index])
    x[train_index] = scaler.transform(x[train_index])
    x[test_index] = scaler.transform(x[test_index])

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

    # clf = GridSearchCV(svm.LinearSVC(C=0.01),tuned_param, cv=cv, scoring=scores[0], verbose=9, n_jobs=5)
    # clf = GridSearchCV(TSVC(calc_probs=calc_probs), tuned_param, cv=cv, scoring=scores[0], verbose=999, n_jobs=5)
    # clf = cross_val_score(TSVC(calc_probs=calc_probs, C=tuned_param['C'][0],
    #     kernel=tuned_param['kernel'][0], gamma=tuned_param.get('gamma', [0])[0]), x, y, cv=cv,
    #     scoring=scores[0], n_jobs=5, verbose=999)
    # ---------------------------------------------- t=TSVC(C=1,kernel='rbf')
    # --------- clf = GridSearchCV(t, tuned_parameters, cv=cv, scoring=score)
    # ------------------------------------------------------ x = sp.csr_matrix(x)
    #  ---------------------------------------------------------------
    # ------------------------ clf = svm.LinearSVC(C=0.01, verbose=9, dual=False)
    # ----------------------- #clf = svm.SVC(kernel='linear', C=0.01, verbose=99)
    # ------------------------------------------------ print(str(datetime.now()))
    # ------------------------------------------------------------- clf.fit(x, y)
    # ------------------------------------------------ print(str(datetime.now()))


def print_results(clf, x_test=None, y_test=None, calc_probs=False, path=None, time=None, cv_scores=None, mini_path='', cv_mean_score=0):
    # save(clf, os.path.join(DUMP_FOLDER, 'Est.pkl'))
    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_estimator_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # for params, mean_score, scores in clf.grid_scores_:
    #      print("%0.3f (+/-%0.03f) for %r"
    #       % (mean_score, scores.std() / 2, params))
    #  print()
    #  #
    #  print("Detailed classification report:")
    #  print()
    #  print("The model is trained on the full development set.")
    #  print("The scores are computed on the full evaluation set.")
    #  print()

    if x_test is not None:
        y_true, y_pred = y_test, clf.predict(x_test)
        if calc_probs:
            save(y_pred, os.path.join(mini_path,'YPRED_c_{}_kernel_{}_gamma_{}.pkl'.format(clf.C, clf.kernel, clf.gamma)))
            calc_auc(y_pred, y_true, roc_fig_name=os.path.join(mini_path, str(clf)+'.png'))
            score = auc_score(y_pred, y_true)
            y_pred = probs_to_preds(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)

        if cv_mean_score != 0:
            score = cv_mean_score
        # score=abs(cm_normalized[0,0]-cm_normalized[0,1]+cm_normalized[1,1]-cm_normalized[1,0])
        score_str = "{:.3f}".format(score)
        path = path.split('/')[-1]

        directory = os.path.join(DUMP_FOLDER, mini_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save(clf, os.path.join(mini_path, 'CLF_c_{}_kernel_{}_gamma_{}_score_{}.pkl'.format(clf.C, clf.kernel, clf.gamma, score_str)))
    #    save(clf, os.path.join(DUMP_FOLDER, 'Est'+str(score)+'.pkl'))

        report(str(clf), path, score, 'Est'+str(score)+'.pkl', time, cv_scores, cm_normalized)
        # only if calc_probs
        print(classification_report(y_true, y_pred))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def auc_Score(ypred, ytrue):
    # Do something
    # fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred, pos_label=2)
    fpr, tpr, thresholds = roc_curve(ytrue, ypred[:, 1])
    return sklearn.metrics.auc(fpr, tpr)


def fpr_tpr_auc_calc(ypred, ytrue):
    fpr, tpr, _ = roc_curve(ytrue, ypred[:, 1])
    return fpr, tpr, sklearn.metrics.auc(fpr, tpr)


def micro_fpr_tpr_auc_calc(ypred, ytrue):
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(ypred.ravel(), ytrue.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def gmean_score(ytrue, ypred):
    rates = calc_rates(ytrue, ypred)
    return gmean_score_from_rates(rates)


def calc_rates(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred, [0, 1])
    r1 = (confusion_mat[0, 0] / float(np.sum(confusion_mat[0, :])))
    r2 = (confusion_mat[1, 1] / float(np.sum(confusion_mat[1, :])))
    return r1, r2


def gmean_score_from_rates(rates):
    return np.sqrt(rates[0]*rates[1])


def shuffle(x):  # seed=13
    idx = np.arange(x.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    x_shuf = x[idx]
    return x_shuf, idx


def check_if_params_were_calculated(dump_folder, tuned_params):
    file_to_create = os.path.join(dump_folder, 'YPRED_c_{}_kernel_{}_gamma_{}.pkl'.format(tuned_params.get('C')[0], tuned_params.get('kernel')[0], tuned_params.get('gamma')[0]))
    file_exists = os.path.isfile(file_to_create)
    if not file_exists:
        pass
    return file_exists


def num_of_line_in_summary_file():
    with open(os.path.join(DUMP_FOLDER, 'summary.txt')) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


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




'''
        self._impl = impl
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
'''


def save(obj, file_name):
    if SAVE:
        full_file_name = os.path.join(CURRENT_DUMP_FOLDER, file_name)
        with open(full_file_name, 'w') as pklFile:
            pickle.dump(obj, pklFile)


def probs_to_preds(probs):
    return np.array([0 if p[0] > 0.5 else 1 for p in probs])


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
        parts_of_path = feature_folder.split('/')
        dump_path = os.path.join(DUMP_FOLDER, parts_of_path[-1])
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        else:
            continue
        x, y, x_test, y_test, inds = load_data(feature_folder)
        print(feature_folder)
        folds_num = 5

        run(x, y, x_test, y_test, folds_num, feature_folder, inds, 5)
    print('finish!')


if __name__ == '__main__':
    cProfile.run("print go(); print")