'''
Created on Mar 30, 2015

@author: ohadfel
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np, h5py 


# import some data to play with
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features.
#Y = iris.target

f = h5py.File('/home/ohadfel/Desktop/4ohad/Xtrain.mat','r') 
data = f.get('XTrain') 
X = np.array(data) # For converting to numpy array
X=X.T

f = h5py.File('/home/ohadfel/Desktop/4ohad/Ytrain.mat','r') 
data = f.get('YTrain') 
Y = np.array(data) # For converting to numpy array
Y=np.squeeze(Y)
Y=Y.T

f = h5py.File('/home/ohadfel/Desktop/4ohad/Xtest.mat','r') 
data = f.get('XTest') 
Xtest = np.array(data) # For converting to numpy array
Xtest=Xtest.T

f = h5py.File('/home/ohadfel/Desktop/4ohad/Ytest.mat','r') 
data = f.get('YTest') 
Ytest = np.array(data) # For converting to numpy array
Ytest=np.squeeze(Y)
Ytest=Ytest.T




x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
colors = ['r', 'g', 'b']
c = [colors[int(y)] for y in np.squeeze(Y)]
ax.scatter(X_reduced[:, 0], X_reduced[:, 1],X_reduced[:, 2], c=c)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
# Percentage of variance explained for each components
print('explained variance ratio (first two components): {}'.format(
      sum(pca.explained_variance_ratio_)))