# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 08:54:14 2019

@author: balam
"""

import csv
import matplotlib.pyplot as plt

def readTrainFile(fileNameTrain, plot = False):
    with open(fileNameTrain) as file:
        X = []
        Y = []
        rows = csv.reader(file, delimiter = ",")
        for row in rows:
            X.append(float(row[0]))
            Y.append(float(row[1]))
        if plot:
            plt.figure()
            plt.plot(X,Y,'o')
            plt.show()
        return X, Y
    
def readTestFile(fileNameTest, plot = False):
    with open(fileNameTest) as file:
        X = []
        rows = csv.reader(file, delimiter = ",")
        for row in rows:
            X.append(float(row[0]))
        if plot:
            plt.figure()
            plt.plot(X,[0]*len(X),'o')
            plt.show()
        return X

if __name__ == "__main__":
    fileNameTrain = "Data/problem4a_train.csv"
    fileNameTest = "Data/problem4a_test.csv"
    readTrainFile(fileNameTrain, True)
    readTestFile(fileNameTest, True)