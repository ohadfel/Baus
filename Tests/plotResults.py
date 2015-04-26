__author__ = 'ohadfel'
import pickle
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
import os
import h5py

def loadData(path):
    f = h5py.File(path+'/Xtest.mat', 'r')
    data = f.get('XTest')
    Xtest = np.array(data)  # For converting to numpy array
    Xtest = Xtest.T

    f = h5py.File(path+'/Ytest.mat', 'r')
    data = f.get('YTest')
    Ytest = np.array(data)  # For converting to numpy array
    Ytest = np.squeeze(Ytest)
    Ytest = Ytest.T

    return(Xtest,Ytest)


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
###############################################################################
# Data IO and generation

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X, y = X[y != 2], y[y != 2]
# n_samples, n_features = X.shape
#
# # Add noisy features
# random_state = np.random.RandomState(0)
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
# ###############################################################################
# # Classification and ROC analysis
#
# # Run classifier with cross-validation and plot ROC curves
# cv = StratifiedKFold(y, n_folds=6)
# classifier = svm.SVC(kernel='linear', probability=True,
#                      random_state=random_state)
#
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []
#
# for i, (train, test) in enumerate(cv):
#     probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
# mean_tpr /= len(cv)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


if __name__ == '__main__':

    basePath = '/home/ohadfel/Copy/Baus'
    path = os.path.join(basePath, 'Pre', 'data1')
    x_test, y_test = loadData(path)

    directory = '/media/ohadfel/New Volume/Baus-Dumps/data1'
    differentDataPaths = [x[2] for x in os.walk(directory)]
    differentDataPaths = differentDataPaths[0]
    C = [1e-2, 1e-1, 1e0, 1e1]
    for plotC in C:
        for curFileStr in differentDataPaths:
            if curFileStr[0] == 'Y':
                fileName = os.path.join(directory, curFileStr)
                q = curFileStr.split('_')
                if float(q[2]) == plotC:
                    with open(fileName, 'r') as pklFile:
                        probas_ = pickle.load(pklFile)
                        # Compute ROC curve and area the curve
                        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
                        roc_auc = auc(fpr, tpr)

                        # accuracy=
                        plt.plot(fpr, tpr, lw=5*float(roc_auc), label='C=%s ,gamma=%s (area = %0.2f)' % (q[2], q[6][:-4], roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve with RBF kernel')
        plt.legend(loc="lower right")
        plt.show()

print('OK')