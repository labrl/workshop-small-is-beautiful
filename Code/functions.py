# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:35:40 2020

@authors: Marc Lanovaz
"""
#Import packages and functions
import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

lm = LinearRegression()
standard_scaler = preprocessing.StandardScaler()

#Set random seed for replicability
np.random.seed(48151)

#Create one series
def create_ABseries(a, t, constant, nb_pointsA, nb_pointsB, SMD):  
    
    #Compute the total number of points
    nb_points = nb_pointsA + nb_pointsB
    
    #Start with empty series
    data_series = []
    
    #For number of points generate error term
    for i in range(nb_points):
        
        #To deal with first point only
        if not data_series:
            error = np.random.normal()
            data_series.append(error)
        
        #Points other than first - Add autocorrelation
        else: 
            error = a*(data_series[i-1])+np.random.normal()
            data_series.append(error)
    
    #Add trend
    middle_point = np.median(range(nb_points))

    for i in range(nb_points):
        diff = i - middle_point
        data_series[i] = data_series[i] + diff*tan(radians(t))
    
    #Add constant  
    data_series = [x+constant for x in data_series]
    
    #Data labels A and B
    data_labels = ['A'] *nb_pointsA +['B']*nb_pointsB
    
    #Add SMD to each point of phase B
    for j in range(nb_pointsA,nb_points):
        data_series[j] = data_series[j]+ SMD
    
    final_series = np.vstack((data_labels, data_series))
    
    return final_series

#Standardize data so that each graph has a mean of 0 and standard deviation of 1
def extract_features(x):
    
    #Create features list
    features = []
        
    #Identify indices for points of each phase
    indexA = np.where(x[0]== 'A')
    indexB = np.where(x[0]== 'B')
    
    #Transform values to float vectors
    phaseA = (x[1,indexA].astype(float)).flatten()
    phaseB = (x[1,indexB].astype(float)).flatten()
    
    #Compute mean and standard deviation
    overallMean = np.mean(np.hstack((phaseA, phaseB)))
    overallSd = np.std(np.hstack((phaseA, phaseB)))
        
    #Transform points in Phase A to z scores
    phaseA = (phaseA-overallMean)/overallSd
    phaseA[np.isnan(phaseA)] = 0
    
    #Transform points in Phase B to z scores
    phaseB = (phaseB-overallMean)/overallSd
    phaseB[np.isnan(phaseB)] = 0
    
    #Get length of phase A and total number of points
    pointsA = len(phaseA)
    pointsTotal = pointsA + len(phaseB)
    
    #Add mean to features vector
    features.append([np.mean(phaseA), np.mean(phaseB)])
    
    #Add standard deviation to features vector
    features.append([np.std(phaseA),np.std(phaseB)])
    
    #Compute and append slope and intercept for Phase A
    lm1 = LinearRegression().fit(np.array(range(pointsA)).reshape(-1,1), 
                    np.expand_dims(phaseA, axis =1))
    features.append([float(lm1.intercept_), float(lm1.coef_)])

    #Compute and append slope and intercept for Phase B
    lm2 = LinearRegression().fit(np.array(range(pointsA,pointsTotal))\
                    .reshape(-1,1), np.expand_dims(phaseB, axis =1))
    features.append([float(lm2.intercept_), float(lm2.coef_)])
    
    #Transform list to vector
    features = np.array(features).flatten()
    
    #Return features vector
    return features

#Graph AB
def ABgraph(series):
    A = np.where(series[0] == 'A')
    B = np.where(series[0] == 'B')
    valuesA = series[1][A].astype(float)
    valuesB = series[1][B].astype(float)
    
    ylim_value = np.min([0,np.min(valuesA)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(A[0]+1, valuesA, 'k', B[0]+1, valuesB, 'k', marker = 's', 
             clip_on=False)
    plt.axvline(x=len(A[0])+0.5, color = 'k')
    plt.xlabel('Session')
    plt.ylabel('Behavior')
    plt.ylim(ylim_value, np.max(series[1].astype(float))*1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


#Function to generate dataset
def generate_dataset():
    #Set values of autocorrelation 'a', trend in degrees 't', number of points in
    #phase A and phase B, and standardized mean difference 'smd'
    a_values = [0,0.2]
    t_values = [0,30]
    constant_values = [4,10]
    pointsa_values = [3,5]
    pointsb_values = [5,10]
    smd_values = [0,0,0,0,0,1,2,3,4,5]

    #Generate 8000 graphs with varying characteristics
    dataset = []
    for i in range(25):
        for a in a_values: 
            for t in t_values: 
                for constant in constant_values:
                    for points_a in pointsa_values:
                        for points_b in pointsb_values:
                            for smd in smd_values: 
                                dataseries = create_ABseries(a, t, constant, 
                                                        points_a, points_b, smd)
                                dataset.append([dataseries, [a,t,
                                            constant, points_a, points_b, smd]])
        
    #Randomize order of series
    shuffled_order = np.random.choice(range(8000), 8000, replace = False)
    shuffled_dataset = []
    for i in shuffled_order:
        shuffled_dataset.append(dataset[i])

    #Extract features and class labels
    x = np.empty((0,8))
    y = np.empty((0,))

    for i in range(len(shuffled_dataset)):
        series = shuffled_dataset[i][0]
        features = extract_features(series).reshape(1,-1)
        x = np.vstack((x, features))
        y = np.hstack((y, shuffled_dataset[i][1][5]))

    y[y>0] = 1

    return x, y