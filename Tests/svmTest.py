'''
Created on Mar 23, 2015

@author: ohadfel
'''
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp
import h5py
import pickle
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
import cProfile

import numpy as np
#from sklearn.metrics.metrics import auc_score
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn import datasets, svm
from datetime import datetime
import time
from crosVal import IndicesKFold
import utils
from collections import Counter
import socket
from sklearn.svm import LinearSVC

# np.logspace(-2, 3, 6)

# DUMP_FOLDER = '/home/ohadfel/Copy/Baus/dumps'
DUMP_FOLDER = 'C:\Users\Ohad\Copy\Baus\dumps'

# DUMP_FOLDER = '/media/ohadfel/New Volume/Results'

# tuned_parameters = [{'kernel': ['linear'], 'C': [0.01,0.1,1, 10, 100, 1000]},{'kernel': ['rbf'], 'gamma': [0,1e-3, 1e-4],'C': [0.01,0.1,1, 10, 100, 1000]}]
# tuned_parameters = [{'kernel': ['linear'], 'C': [0.01]},{'kernel': ['linear'], 'C': [0.1]},{'kernel': ['linear'], 'C': [1]}, \
#                     {'kernel': ['rbf'], 'gamma': [0],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.01]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [0],'C': [0.1]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.1]},{'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.1]},\
#                     {'kernel': ['rbf'], 'gamma': [0],'C': [1]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [1]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-4],'C': [1]}]
tuned_parameters = [{'kernel': ['linear'], 'C': [0.01]},{'kernel': ['linear'], 'C': [0.1]},{'kernel': ['linear'], 'C': [1]}, \
                    {'kernel': ['rbf'], 'gamma': [0],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.01]},\
                    {'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [0],'C': [0.1]},\
                    {'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.1]},{'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.1]},\
                    {'kernel': ['rbf'], 'gamma': [0],'C': [1]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [1]},\
                    {'kernel': ['rbf'], 'gamma': [1e-4],'C': [1]}]

# tuned_parameters = [{'kernel': ['linear'], 'C': [0.1]},{'kernel': ['linear'], 'C': [1]}, \
#                     {'kernel': ['rbf'], 'gamma': [0],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.01]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.01]},{'kernel': ['rbf'], 'gamma': [0],'C': [0.1]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-3],'C': [0.1]},{'kernel': ['rbf'], 'gamma': [1e-4],'C': [0.1]},\
#                     {'kernel': ['rbf'], 'gamma': [0],'C': [1]},{'kernel': ['rbf'], 'gamma': [1e-3],'C': [1]},\
#                     {'kernel': ['rbf'], 'gamma': [1e-4],'C': [1]}]'''
# tuned_parameters = [{'kernel': ['linear'], 'C': [0.1]}] '''

#tuned_parameters = [{'C': [0.01],'verbose':[0],'dual':[False]},{'C': [0.1],'verbose':[0],'dual':[False]},{'C': [1],'verbose':[0],'dual':[False]},{'C': [10],'verbose':[0],'dual':[False]}]
#tuned_parameters = [{'kernel': ['linear'], 'C': [0.01]},{'kernel': ['linear'], 'C': [0.1]},{'kernel': ['linear'], 'C': [1]},\
 #                   {'kernel': ['linear'], 'C': [1]},{'kernel': ['linear'], 'C': [10]}]'''

class Timer(object):
    def __init__(self, name=None,doPrint=True):
        self.name = name
        self.doPrint = doPrint

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.doPrint:
            if self.name:
                print '[%s]' % self.name,
            calc=(time.time() - self.tstart)
            print 'Elapsed: '+str(calc)
        return calc


def run(X, y,XTest,yTest, foldsNum,path,inds, jobsNum=6, calcProbs=True):
#    y = y.astype(np.int)
#    yTest = yTest.astype(np.int)

    # heldoutSize = 1./foldsNum
    # X, idx = shuffle(X)
    # y = y[idx]
    # XTest, idx = shuffle(XTest)
    # yTest = yTest[idx]

    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # XTest = scaler.transform(XTest)

    # X = X[:5000, :]
    # y = y[:5000]
    #---------------------------------------------------- XTest = XTest[:500, :]
    #------------------------------------------------------- yTest = yTest[:500]

    # cv = StratifiedShuffleSplit(y, foldsNum, heldoutSize, random_state=0)
    cv = IndicesKFold(inds, 5)#, 4000, 1000)
    #-------------------------------- scores = ['roc_auc','precision', 'recall']
    #-------------- aucScoreFunc = make_scorer(aucScore, greater_is_better=True)
    scores = ['roc_auc'] #[aucScoreFunc] # 'roc_auc'
    #---------------------------------------------------- scores = ['precision']
    calcProbs = True

    print('Start the grid search')
    t = time.time()
    for tuned_param in tuned_parameters:
        params = []
        for fold_num, (train_index, test_index) in enumerate(cv):
            params.append((X, y, train_index, test_index, tuned_param, fold_num, calcProbs))

        if (jobsNum == 1):
            mapResults = [calc_cv_scores(p) for p in params]  # For debugging
        else:
            mapResults = utils.parmap(calc_cv_scores, params, jobsNum)

        cv_scores = np.array([score for (clf, score) in mapResults][:len(cv)])
        print(cv_scores)
        clf = mapResults[0][0]

        elapsed = time.time() - t
        print('Request took '+str(elapsed)+' sec.')
        print(str(datetime.now()))

        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        XTest = scaler.transform(XTest)

        printResults(clf, XTest, yTest, calcProbs, path)


def calc_cv_scores(p):
    t = time.time()
    X, y, train_index, test_index, tuned_param, fold_num, calcProbs = p

    scaler = preprocessing.StandardScaler().fit(X[train_index])
    X[train_index] = scaler.transform(X[train_index])
    X[test_index] = scaler.transform(X[test_index])

    clf = TSVC(C=tuned_param['C'][0], kernel=tuned_param['kernel'][0], gamma=tuned_param.get('gamma', [0])[0], calcProbs=calcProbs)
    print(fold_num, str(datetime.now()))
    print('number of train samples'+str(X[train_index].shape), 'number of test samples'+str(X[test_index].shape))
    print(Counter(y))
    t = time.time()
    clf.fit(X[train_index], y[train_index])
    ypred = clf.predict(X[test_index])
    score = aucScore(ypred, y[test_index])
    elapsed = time.time() - t
    print(tuned_param)
    print('Request took '+str(elapsed)+' sec.')
    return clf, score


    #clf = GridSearchCV(svm.LinearSVC(C=0.01),tuned_param, cv=cv, scoring=scores[0], verbose=9, n_jobs=5)
    # clf = GridSearchCV(TSVC(calcProbs=calcProbs), tuned_param, cv=cv, scoring=scores[0], verbose=999, n_jobs=5)
    # clf = cross_val_score(TSVC(calcProbs=calcProbs, C=tuned_param['C'][0],
    #     kernel=tuned_param['kernel'][0], gamma=tuned_param.get('gamma', [0])[0]), X, y, cv=cv,
    #     scoring=scores[0], n_jobs=5, verbose=999)
    #---------------------------------------------- t=TSVC(C=1,kernel='rbf')
    #--------- clf = GridSearchCV(t, tuned_parameters, cv=cv, scoring=score)

    #------------------------------------------------------ X = sp.csr_matrix(X)
#---------------XSW21qaz
# ---------------------------------------------------------------
    #------------------------ clf = svm.LinearSVC(C=0.01, verbose=9, dual=False)
    #----------------------- #clf = svm.SVC(kernel='linear', C=0.01, verbose=99)
    #------------------------------------------------ print(str(datetime.now()))
    #------------------------------------------------------------- clf.fit(X, y)
    #------------------------------------------------ print(str(datetime.now()))

def printResults(clf, xtest=None, ytest=None, calcProbs=False,path=None,time=None):
   # save(clf, os.path.join(DUMP_FOLDER, 'Est.pkl'))
   #  print("Best parameters set found on development set:")
   #  print()
   #  print(clf.best_estimator_)
   #  print()
   #  print("Grid scores on development set:")
   #  print()
   #  for params, mean_score, scores in clf.grid_scores_:
   #      print("%0.3f (+/-%0.03f) for %r"
   #            % (mean_score, scores.std() / 2, params))
   #  print()
   #
   #  print("Detailed classification report:")
   #  print()
   #  print("The model is trained on the full development set.")
   #  print("The scores are computed on the full evaluation set.")
   #  print()


    if (not xtest is None):
        y_true, y_pred = ytest, clf.predict(xtest)
        if (calcProbs):
            calcAUC(y_pred, y_true,ROCFigName=DUMP_FOLDER+'/'+str(clf)+'.png')
            score=aucScore(y_pred, y_true)
            y_pred = probsToPreds(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)
        #score=abs(cm_normalized[0,0]-cm_normalized[0,1]+cm_normalized[1,1]-cm_normalized[1,0])
        scoreStr="{:.3f}".format(score)
        path = path.split('/')[-1]
        save(clf, os.path.join(DUMP_FOLDER, 'c_{}_kernel_{}_gamma_{}_score_{}_path_{}.pkl'.format(clf.C, clf.kernel, clf.gamma, scoreStr, path[:-4])))
    #    save(clf, os.path.join(DUMP_FOLDER, 'Est'+str(score)+'.pkl'))

        report(str(clf),path,score,'Est'+str(score)+'.pkl',time)
        # only if calcProbs
        print(classification_report(y_true, y_pred))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def aucScore(ypred, ytrue):
    # Do something
    #fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred, pos_label=2)
    fpr, tpr, thresholds = roc_curve(ytrue, ypred[:, 1])
    return sklearn.metrics.auc(fpr, tpr)

def gmeanScore(ytrue, ypred):
    rates = calcRates(ytrue, ypred)
    return gmeanScoreFromRates(rates)


def calcRates(ytrue, ypred):
    conMat = confusion_matrix(ytrue, ypred, [0, 1])
    r1 = (conMat[0, 0] / float(np.sum(conMat[0, :])))
    r2 = (conMat[1, 1] / float(np.sum(conMat[1, :])))
    return r1,r2


def gmeanScoreFromRates(rates):
    return np.sqrt(rates[0]*rates[1])


def shuffle(x):  # seed=13
    idx = np.arange(x.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    xshuf = x[idx]
    return (xshuf, idx)


class TSVC(SVC):

    def __init__(self, C=1, kernel='rbf', gamma=0, calcProbs=True):
        super(TSVC, self).__init__(C=C, kernel=kernel, gamma=gamma, probability=True)
        self.calcProbs = calcProbs
        #----------------------------------------------- print("in constructor")
        # print('C='+str(C)+', kernel='+kernel+', gamma='+str(gamma)+', probability=True')

    def fit(self, X, y, doShuffle=True):
#         if (doShuffle):
#             (X, idx) = shuffle(X)
#             y = y[idx]
#         self.scaler = preprocessing.StandardScaler().fit(X)
#         X = self.scaler.transform(X)
        super(TSVC, self).fit(X, y)
        return self

    def predict(self, X):
        print('predict!')
#         X = self.scaler.transform(X)
        if (self.calcProbs):
            probs = super(TSVC, self).predict_proba(X)
        else:
            probs = super(TSVC, self).predict(X)
#         score = roc_auc_score(self.ytrue, probs[:,1])
        # save
#         dumpFile = os.path.join(DUMP_FOLDER, 's_{}_{}_{}.pkl'.format(self.kernel, self.C, self.gamma))
#         print('save to {}'.format(dumpFile))
#         save(self, dumpFile)
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

def save(obj, fileName):
    with open(fileName, 'w') as pklFile:
        pickle.dump(obj, pklFile)

def probsToPreds(probs):
    return np.array([0 if p[0]>0.5 else 1 for p in probs])


def loadData(path):
    f = h5py.File(path+'/Xtrain.mat','r')
    data = f.get('XTrain')
    X = np.array(data) # For converting to numpy array
    X=X.T

    f = h5py.File(path+'/Ytrain.mat','r')
    data = f.get('YTrain')
    Y = np.array(data) # For converting to numpy array
    Y=np.squeeze(Y)
    y=Y.T

    f = h5py.File(path+'/Xtest.mat','r')
    data = f.get('XTest')
    Xtest = np.array(data) # For converting to numpy array
    Xtest=Xtest.T

    f = h5py.File(path+'/Ytest.mat','r')
    data = f.get('YTest')
    Ytest = np.array(data) # For converting to numpy array
    Ytest=np.squeeze(Ytest)
    Ytest=Ytest.T

    return(X,y,Xtest,Ytest)

def report(params,path,score,dumpFileName,time):
    # AUC Score=88,
    f=open(DUMP_FOLDER+'/log.txt','a')
    f.seek(0) #get to the first position
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCORE='+str(score)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    line='params '+params+' dataset '+path+' score '+str(score)+' dump File Name '+dumpFileName+' Time was '+str(time)+ "\n"
    f.write(line)
    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    f.close()

def calcAUC(probs, y, doPlot=True,ROCFigName=''):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    if (ROCFigName!=''):
        doPlot=False
    return plotROC(mean_tpr, mean_fpr, doPlot, fileName=ROCFigName)


def plotROC(mean_tpr, mean_fpr, doPlot, lenCV=1, fileName=''):
    if (doPlot):
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

    mean_tpr /= float(lenCV)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #if (doPlot):
    plt.ion()
    plt.plot(mean_fpr, mean_tpr, 'r-', label='Mean ROC (area = %0.2f)' % mean_auc, lw=3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if (fileName!=''):
        plt.savefig(fileName)

    if (doPlot):
        plt.show()
    else:
        plt.close()
    return mean_auc, mean_tpr, mean_fpr


def go():
    # directory='/home/ohadfel/Desktop/4ohad/Pre'
    # differentDataPaths=[x[0] for x in os.walk(directory)]
    # differentDataPaths=differentDataPaths[1:]
    if (socket.gethostname()=='Ohad-PC'):
	    basePath='C:\Users\Ohad\Copy\Baus'
    else:
	    basePath='/home/ohadfel/Copy/Baus'

    path = os.path.join(basePath, 'Code', 'matlab', 'inds.mat')

    f = h5py.File(path, 'r')
    inds= f.get('patternsCrossValInd')
    inds = np.array(inds, dtype=np.int)


    #path='/home/ohadfel/Desktop/4ohad/Last_change'
    #path='/home/ohadfel/Copy/Baus/Pre/data1'
    path = os.path.join(basePath, 'Pre', 'data1')
    X,y,Xtest,Ytest=loadData(path)
    print(os.path.dirname(os.path.abspath(__file__)))
    foldsNum = 5

    run(X, y, Xtest, Ytest, foldsNum, path, inds, 2)
    print('finish!')


if __name__ == '__main__':
    cProfile.run("print go(); print")