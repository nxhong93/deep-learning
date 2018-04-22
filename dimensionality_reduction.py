import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)



pca = PCA(n_components = 2)
x2D = pca.fit_transform(X)
print (x2D[0])

pca1 = PCA(n_components=2, svd_solver="randomized")
x2D1 = pca1.fit_transform(X)
print (x2D1[0])

pca2 = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.04)
x2D2 = pca2.fit_transform(X)
print (x2D2[0])

pca3 = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
x2D3 = pca2.fit_transform(X)
print (x2D3[0])

pca4 = KernelPCA(n_components = 2, kernel="linear")
x2D4 = pca4.fit_transform(X)
print (x2D4[0])

lle = LocallyLinearEmbedding(n_components = 2, n_neighbors = 10, random_state = 40)
x2D5 = lle.fit_transform(X)
print (x2D5[0])
