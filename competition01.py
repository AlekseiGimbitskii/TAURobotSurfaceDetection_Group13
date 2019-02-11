# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:59:11 2019

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier


#------------feature extraction functions-------------
def stupidResize(X_train):
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])    
    #print(X_train.shape)
    return X_train

def average(X_train):
    X_train = np.mean(X_train,axis=2)
    #print(X_train.shape)    
    return X_train

def averageAndDeviation(X_train):
    average = np.mean(X_train, axis=2)
    std = np.std(X_train, axis=2)
    X_train = np.concatenate((average,std), axis=1)
    #print(X_train.shape)    
    return dataNormalization(X_train)


def filteredFeature(X_train):
    X_train = X_train[:,[0,1,2,3],:]
    average = np.mean(X_train, axis=2)
    std = np.std(X_train, axis=2)
    X_train = np.concatenate((average,std), axis=1)   
       
    #print(X_train.shape)
    return X_train     

def vectorLengthsFeature(X_train):
    #Orientation = np.sqrt(np.power((X_train[:,[0],:]/X_train[:,[3],:]),2)+np.power((X_train[:,[1],:]/X_train[:,[3],:]),2)+np.power((X_train[:,[2],:]/X_train[:,[3],:]),2))
    AngularVelocity = np.sqrt(np.power(X_train[:,[4],:],2)+np.power(X_train[:,[5],:],2)+np.power(X_train[:,[6],:],2))
    LinearAcceleration = np.sqrt(np.power(X_train[:,[7],:],2)+np.power(X_train[:,[8],:],2)+np.power(X_train[:,[9],:],2))
    X_train = np.concatenate((X_train[:,[0],:],X_train[:,[1],:],X_train[:,[2],:],X_train[:,[3],:],AngularVelocity,LinearAcceleration ), axis=1) 

    average = np.mean(X_train, axis=2)
    std = np.std(X_train, axis=2)
    X_train = np.concatenate((average,std), axis=1)  
    
    #print(X_train.shape)
    return dataNormalization(X_train)

def dataNormalization(X_train):
    for i in range(X_train.shape[1]):
        X_train[:, i] = X_train[:, i] / np.max(X_train[:, i])
    return X_train
#-----------------load data---------------------
X_test_submission = np.load("X_test_kaggle.npy")
X_train = np.load("X_train_kaggle.npy")
y_train = np.loadtxt("y_train_final_kaggle.csv", dtype = np.str , delimiter = ',', usecols=(0,1), unpack=False)
y_train = y_train[:,1] 
#print(y_train.shape)
#y_train[:,0] = y_train[:,0].astype(np.int)


#--------------create an indexes from class names------
le = LabelEncoder()
le.fit(y_train)
classes = le.transform(y_train)
y_train = np.column_stack((y_train, classes)) # stack stings and their indexes
#print(y_train[:5,:5])

#--------------split the data for train ant tes--------------------------
#they write something about sklearn.model_selection.ShuffleSplit but I
#dont understand what they want, seems like they already shuffled the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train[:,0], test_size=0.2)


#------------------classification with different models-----------
modelList = []
modelList.append(KNeighborsClassifier(n_neighbors = 1))
modelList.append(LinearDiscriminantAnalysis())
modelList.append(SVC())
modelList.append(LogisticRegression())
modelList.append(RandomForestClassifier())
modelList.append(RandomForestClassifier(n_estimators = 100))
modelList.append(ExtraTreesClassifier(n_estimators = 1000))
#modelList.append(AdaBoostRegressor(n_estimators = 100))
#modelList.append(GradientBoostingClassifier(n_estimators = 100)) #bad performance

global_max_score = []
global_model = []
for model in modelList:
    max_score = []
    modelName = type(model).__name__
    print(f'Name model: {modelName}')
    
    model.fit(stupidResize(X_train), y_train)
    score = accuracy_score(y_test, model.predict(stupidResize(X_test)))    
    print(f'Feature: resize , Score: {score}')
    max_score.append(score)
    model.fit(average(X_train), y_train)
    score = accuracy_score(y_test, model.predict(average(X_test)))    
    print(f'Feature: averaging , Score: {score}')
    max_score.append(score)
    model.fit(averageAndDeviation(X_train), y_train)
    score = accuracy_score(y_test, model.predict(averageAndDeviation(X_test)))    
    print(f'Feature: average and std_deviation , Score: {score}')
    max_score.append(score)
    model.fit(filteredFeature(X_train), y_train)
    score = accuracy_score(y_test, model.predict(filteredFeature(X_test)))    
    print(f'Feature: Filtered , Score: {score}')
    max_score.append(score)
    model.fit(vectorLengthsFeature(X_train), y_train)
    score = accuracy_score(y_test, model.predict(vectorLengthsFeature(X_test)))    
    print(f'Feature: vectorLengths , Score: {score}')
    max_score.append(score)
    print(f'Name model MAX SCORE: {max(max_score)}, model_id: {np.argmax(max_score)}')
    global_max_score.append(max(max_score))
    global_model.append(np.argmax(max_score))

print(f'MAX SCORE: {max(global_max_score)}, ID GLOBAL: {np.argmax(global_max_score)}, model_ids: {global_model}')

#-------------create submission------------
model = ExtraTreesClassifier(n_estimators = 1000)
model.fit(averageAndDeviation(X_train), y_train)
y_pred = model.predict(averageAndDeviation(X_test_submission))
#labels = list(le.inverse_transform(y_pred))
labels = y_pred


with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))

'''
stupidResize(X_train)
average(X_train)
averageAndDeviation(X_train)
spectrogramFeature(X_train)'''


