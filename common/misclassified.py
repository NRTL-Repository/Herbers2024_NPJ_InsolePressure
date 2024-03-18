# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:14:03 2023

@author: c_pia
"""

import logging


__all__ = ["find_common_misclassified"]

logger = logging.getLogger(__name__)


def find_common_misclassified(clfs, y_working):
    '''
    function: find_common_misclassified
    
    description: finds the samples which were commonly misclassified
        
    
    parameters: 
        clfs:  list of MyModel objects. each My Model object will need
        to have clf.mis_class initialized

    
    returns:
        common_mis_idx: indices of samples which were commonly misclassified
        mis_count: count of individuals who were commonly misclassified


    '''
    dicts = {}
    keys = range(y_working.shape[0])
    
    for i in keys:
        dicts[i] = 0
        
    for clf in clfs:
        for j in clf.mis_class:
            dicts[j] = dicts[j] + 1
            
            
    common_mis_idx = [k for k, v in dicts.items() if v > 2]
    mis_count = dicts
    
    return common_mis_idx, mis_count