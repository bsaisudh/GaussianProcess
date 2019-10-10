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
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ
from sklearn.gaussian_process.kernels import ExpSineSquared as ESS

xTrain, yTrain = readTrainFile("Data/problem4a_train.csv")
xTest = readTestFile("Data/problem4a_test.csv")

xTrain = np.atleast_2d(xTrain)
xTest = np.atleast_2d(xTest)

C1 = C(1.0, (1e-3, 1e3))
C2 = C(0.5, (1e-3, 1e3))
RBF1 = RBF(10, (1e-2, 1e2))
RBF2 = RBF(0.5, (1e-2, 1e2))
RQ1 = RQ(10, 0.5 ,(1e-2, 1e2))
ESS1 = ESS(1.0, 1.0, (1e-05, 100000.0), (1e-05, 100000.0))

kernel1 = C1 * RBF1 + C2
kernel2 = C1 * RBF1 + RBF2
kernel3 = C1 * RQ1 + RBF2
kernel4 = C1 * ESS1 + RBF2

GP = []
for kernel in [kernel1, kernel2, kernel3, kernel4]:
    gp = GaussianProcessRegressor(kernel=kernel, #alpha=0.5 ** 5,
                              n_restarts_optimizer=10)
    gp.fit(xTrain, yTrain)
    GP.append(gp)
    
f, axs = plt.subplots(2,2)
for ndx, gp, ax in zip([1,2,3,4], GP, [[0,0],[0,1],[1,0],[1,1]]):
    y_pred, sigma = gp.predict(xTest, return_std=True)
    axs[ax[0],ax[1]].plot(xTrain, yTrain, 'r:', label="sine(3x)")
    axs[ax[0],ax[1]].errorbar(xTrain.ravel(), yTrain, 0, fmt='r.', markersize=10, label='Observations')
    axs[ax[0],ax[1]].plot(xTest, y_pred, 'b-', label='Prediction')
    axs[ax[0],ax[1]].fill(np.concatenate([xTest, xTest[::-1]]),
             np.concatenate([y_pred - 1.95 * sigma,
                            (y_pred + 1.95 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    axs[ax[0],ax[1]].set_xlabel('x')
    axs[ax[0],ax[1]].set_ylabel('sin(3x)')
    axs[ax[0],ax[1]].legend(loc='upper left')
    axs[ax[0],ax[1]].set_title(f"Kernel - {ndx}")

plt.show()