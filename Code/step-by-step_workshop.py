# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:20:31 2023

@author: Marc Lanovaz
"""

#Load necessary packages and functions
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from functions import create_ABseries, extract_features, ABgraph, \
    generate_dataset

#Create one graph (autocorrelation, trend, constant, pointsA, pointsB, SMD)
myfirstgraph = create_ABseries(0, 0, 10, 5, 10, 3)

#Graph the graph
ABgraph(myfirstgraph)

#Extract features
myfirstfeatures = extract_features(myfirstgraph)

#Set random seed for replicability
np.random.seed(48151)

#Generate dataset with 1,024 graphs
x, y = generate_dataset()

#Split data into training and test sets
x_train, x_test, y_train, y_test =\
    train_test_split(x, y, test_size = 0.20, random_state = 48151)

#Support Vector Classifier

##Set hyperparameters
svc = SVC(class_weight = {0:1, 1:0.5})

##Fit model
svc.fit(x_train, y_train)

##Test model
svc.score(x_test, y_test)

##Check Type I error and power
predictions = svc.predict(x_test)

idx0, = np.where(y_test==0)
idx1, = np.where(y_test==1)

##Type I error rate
np.sum(predictions[idx0])/len(idx0)

##Power
np.sum(predictions[idx1])/len(idx1)

