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
            X.append( [ float(i) for i in row[0:-1] ] )
            Y.append(float(row[-1]))
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
            X.append( [ float(i) for i in row ] )
        if plot:
            plt.figure()
            plt.plot(X,[0]*len(X),'o')
            plt.show()
        return X

def readSolutionFile(fileNameTest, plot = False):
    with open(fileNameTest) as file:
        Y = []
        rows = csv.reader(file, delimiter = ",")
        for row in rows:
            Y.append( [ float(i) for i in row ] )
        if plot:
            plt.figure()
            plt.plot(Y,[0]*len(Y),'o')
            plt.show()
        return Y

if __name__ == "__main__":
    fileNameTrain = "Data/problem4b_train.csv"
    fileNameTest = "Data/problem4b_test.csv"
    fileNameSolution = "Data/problem4b_sol.csv"
    xTrain,yTrain = readTrainFile(fileNameTrain, True)
    xTest = readTestFile(fileNameTest, True)
    yTest = readTestFile(fileNameSolution, True)
