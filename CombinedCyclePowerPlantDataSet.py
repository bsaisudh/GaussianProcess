# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:16:27 2019

@author: balam
"""

from DataPreparation import readTestFile
from DataPreparation import readTrainFile

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

xTrain, yTrain = readTrainFile("Data/problem4b_train.csv")
xTest = readTestFile("Data/problem4b_test.csv")

xTrain = np.atleast_2d(xTrain)
xTest = np.atleast_2d(xTest)

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + RBF(20, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.5 ** 5,
                              n_restarts_optimizer=10)
gp.fit(xTrain, yTrain)
y_pred, sigma = gp.predict(xTest, return_std=True)

plt.figure()
plt.plot(xTrain[0], yTrain, 'r:', label="sine(3x)")
plt.errorbar(xTrain[0].ravel(), yTrain, 0, fmt='r.', markersize=10, label='Observations')
plt.plot(xTest, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([xTest[0], xTest[0][::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('x')
plt.ylabel('sin(3x)')
plt.legend(loc='upper left')

plt.show()