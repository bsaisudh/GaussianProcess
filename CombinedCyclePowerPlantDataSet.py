# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:16:27 2019

@author: balam
"""

from DataPreparation import readTestFile
from DataPreparation import readTrainFile

import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ
from sklearn.gaussian_process.kernels import ExpSineSquared as ESS

xTrain, yTrain = readTrainFile("Data/problem4b_train.csv")
xTest = readTestFile("Data/problem4b_test.csv")

xTrain = np.atleast_2d(xTrain)
xTest = np.atleast_2d(xTest)

C1 = C(1.0, (1e-3, 1e3))
C2 = C(0.5, (1e-3, 1e3))
RBF1 = RBF(10, (1e-2, 1e2))
RBF2 = RBF(5, (1e-2, 1e2))
RBF3 = RBF(2, (1e-2, 1e2))
RBF4 = RBF(1, (1e-2, 1e2))
RBF5 = RBF(0.5, (1e-2, 1e2))
RQ1 = RQ(10, 0.5 ,(1e-2, 1e2))
ESS1 = ESS(1.0, 1.0, (1e-05, 100000.0), (1e-05, 100000.0))

kernel1 = C1 * RBF1 + C2
kernel2 = C1 * RBF1 * RBF2 * RBF3 * RBF4 + RBF5
kernel3 = C1 * RQ1 + RBF2
kernel4 = C1 * ESS1 + RBF2

GP = []
for ndx, kernel in zip([1,2,3,4], [kernel1, kernel2, kernel3, kernel4]):
    t = time.time()
    print('---------------------------------------------------------------------------')
    print(f'time - {t} :: Fitting GP for kernel - {ndx}')
    gp = GaussianProcessRegressor(kernel=kernel, #alpha=0.5 ** 5,
                              n_restarts_optimizer=10)
    gp.fit(xTrain, yTrain)
    GP.append(gp)
    print(f'GP for Kernel - {ndx} Finished :: Elapsed Time - {time.time()-t}')
    print('---------------------------------------------------------------------------')
    
f, axs = plt.subplots(2,2)
Y_PRED = []
SIGMA = []
for ndx, gp, ax in zip([1,2,3,4], GP, [[0,0],[0,1],[1,0],[1,1]]):
    y_pred, sigma = gp.predict(xTest, return_std=True)
    Y_PRED.append(y_pred)
    SIGMA.append(sigma)
    axs[ax[0],ax[1]].plot(xTrain[:,0], yTrain, 'r:', label="sine(3x)")
    axs[ax[0],ax[1]].errorbar(xTrain[:,0].ravel(), yTrain, 0, fmt='r.', markersize=10, label='Observations')
    axs[ax[0],ax[1]].plot(xTest[:,0], y_pred, 'b-', label='Prediction')
    axs[ax[0],ax[1]].fill(np.concatenate([xTest[:,0], xTest[::-1,0]]),
             np.concatenate([y_pred - 1.95 * sigma,
                            (y_pred + 1.95 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    axs[ax[0],ax[1]].set_xlabel('x')
    axs[ax[0],ax[1]].set_ylabel('Data [:,0]')
    axs[ax[0],ax[1]].legend(loc='upper left')
    axs[ax[0],ax[1]].set_title(f"Kernel - {ndx}")
plt.show()