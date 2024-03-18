# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:37:37 2023

@author: c_pia
"""

import logging
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import pandas as pd

__all__ = ["fsfs"]

logger = logging.getLogger(__name__)


def fsfs(clf, X_scaled, y, scoring = 'f1', folds = 5):

    '''
    function: fsfs
    
    description: apply forward sequential feature selection with a chosen model, 
    set of input features, and set of output features.
    
    parameters: 
        X_scaled: set of input features - np.array (N, #of features)
        y: set of output features - np.array (N)
        scoring: optional, method of scoring model performance - string

    returns:
        sfs_results: summary of sequential feature selction and mode performance
        at each iteration

    '''

    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=X_scaled.shape[1],
               forward=True,
               floating=False,
               verbose=2,
               scoring = scoring,
               cv = folds)
    
    # Perform SFFS
    sfs1 = sfs1.fit(X_scaled, y)
    
    sfs_results = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T

    return sfs_results