from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np

dt = load_iris()
print (dt)
X = dt.data[:,2:]
y = dt.target
X_test = np.array([[5, 3.5]])


mod = DecisionTreeClassifier(max_depth=2, random_state=42)
mod.fit(X, y)
print (mod.predict(X_test))
print (mod.predict_proba(X_test))


plt.scatter(X[:,0][y==2],X[:,1][y==2], marker='o', c='b')
plt.scatter(X[:,0][y==1],X[:,1][y==1], marker='x', c='r')
plt.scatter(X[:,0][y==0],X[:,1][y==0], marker='^', c='g')
plt.ylabel('Petal width')
plt.xlabel('Petal length')
plt.show()
