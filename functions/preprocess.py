# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:44:35 2023

@author: c_pia
"""
import logging
import numpy as np
from sklearn import preprocessing

__all__ = ["scale_data"]

logger = logging.getLogger(__name__)

def scale_data(df_X):
    
    '''
    function: scale_data
    
    description: spply a minimum-maximum scaler between 0 and 1 to a set of features
    
    parameters: 
        df_X: features - dataframe of size (n, # of features)

    returns:
        X_scaled: scaled features between 0 and 1 - np.array (N, # of features)

    '''
    
    X = np.asarray(df_X)
    
    scaler = preprocessing.MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    return X_scaled