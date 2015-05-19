__author__ = 'ohadfel'

import numpy as np


class IndicesKFold(object):

    def __init__(self, indices, k, m_train=None, m_test=None):
        self.indices = indices.T
        self.k = k
        self.m_train = m_train
        self.m_test = m_test

    def __iter__(self):
        for k in xrange(1, self.k + 1):
            test_indices = self.indices[self.indices[:, 2] == k, 0]-1
            train_indices = self.indices[self.indices[:, 2] != k, 0]-1
            test_indices = shuffle(test_indices)
            train_indices = shuffle(train_indices)
            m_train = len(train_indices) if self.m_train is None else self.m_train
            m_test = len(test_indices) if self.m_test is None else self.m_test
            test_index = test_indices[:m_test]
            train_index = train_indices[:m_train]
            yield train_index, test_index

    def __len__(self):
        return self.k


# def shuffle(x):  # seed=13
#     idx = np.arange(x.shape[0])
#     np.random.seed(13)
#     np.random.shuffle(idx)
#     return x[idx]
