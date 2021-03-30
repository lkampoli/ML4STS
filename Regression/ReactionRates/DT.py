#!/usr/bin/env python
# coding: utf-8

from sklearn.tree import DecisionTreeRegressor

def est_DT():

   hyperparams = [{'criterion': ('mse', 'friedman_mse', 'mae'), 
                   'splitter': ('best', 'random'),             
                   'max_features': ('auto', 'sqrt', 'log2'),  
   }]
   
   estimator = DecisionTreeRegressor()

   return estimator, hyperparams
