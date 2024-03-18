# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:06:43 2023

@author: c_pia
"""

import numpy as np

import logging

__all__ = ["find_common_features", "find_feature_intersection"]

logger = logging.getLogger(__name__)


def find_common_features(clfs, filtered_feat_names):
    
    '''
    function: find_common_features
    
    description: finds the common features
        
    
    parameters: 
        clfs:  list of MyModel objects. each My Model object will need
        to have clf.feat_idxs initialized
        filtered_feat_names: names of top Shapley featyres
    
    returns:
        common_features_idx: indices of common featurs - list
        common_features: names of common features - np.array

    '''
    dicts = {}
    keys = range(360)
    
    for i in keys:
        dicts[i] = 0
    
    for clf in clfs:
        for j in clf.feat_idxs:
            dicts[j] = dicts[j] + 1
            
            
    common_features_idx = [k for k, v in dicts.items() if v > 2]
    common_features = np.array(filtered_feat_names)[common_features_idx]
    
    return common_features_idx, common_features

def find_feature_intersection(common_features, top_shap):
    '''
    function: find_feature_intersection
    
    description: finds the features which appear as common and top Shapley features
        
    
    parameters: 
        common_features: the names of common features - list
        top_shap: the names of top Shapley features - list

    
    returns:
        res: intersection of common and top Shapley features - list


    '''
    s2 = set(common_features)
    p = []
    for _, shap_feats in top_shap:
        s1 = set(shap_feats)
        check = s1.intersection(s2)
        p.append(list(check))
       
    p_fla = [item for sublist in p for item in sublist]
    
    res = [*set(p_fla)]

    
    return res

