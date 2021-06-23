
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:34:04 2019
Machine learning codes using scikit-learn
Journal of Molecular Graphics and Modelling
Volume 105, June 2021, 107891
@author: user
"""

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import time

matrices = np.load('E:/newmydescriptor.npy')
molden = np.load('E:/homolumost2.npy')
print(matrices.shape)
print(molden.shape)
n_samples = len(matrices)
data = matrices.reshape((n_samples, -1))
print(data.shape)
X = data
y = molden
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20, random_state=1)


from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(activation='tanh', alpha=0, hidden_layer_sizes=(80,20), max_iter=50000, learning_rate_init =0.001,tol=0.0001,learning_rate = 'constant')
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

# from sklearn import neighbors
# knn = neighbors.KNeighborsRegressor(5, 'distance')
# predictions = knn.fit(X_train, y_train).predict(X_test)    
# score = knn.score(X_test, y_test)
# print(score)

# from sklearn.ensemble import RandomForestRegressor
# RF = RandomForestRegressor(max_depth=10, random_state=42,n_estimators=150)
# RF.fit(X_train, y_train)
# predictions = RF.predict(X_test)
# print(RF.score(X_test, y_test))


from sklearn.metrics import mean_absolute_error
score = mlp.score(X_test, y_test)
print(score)
MSE = mean_absolute_error(y_test, predictions, multioutput='uniform_average')
print(MSE)

from matplotlib import pyplot as plt
plt.figure(figsize=(5,5))
t = np.arange(-8.5,5,0.1)
plt.plot(predictions, y_test,'o',t,t,'r',markersize=0.5)
plt.xlabel('Prediction, eV')
plt.ylabel('Actual value, eV')
plt.title('Actual Value against Prediction')
plt.savefig('E:/Graph.png')
