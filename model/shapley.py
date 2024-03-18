# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:41:11 2023

@author: c_pia
"""


import shap
import pandas as pd
import numpy as np
import logging


__all__ = ["shapley"]

logger = logging.getLogger(__name__)

def shapley(X, clfs):
    
    '''
    function: shapley
    
    description: pre-filter features based on ANOVA and f-statistic. allfeatures
    with a p value less than 0.05 are removed
        
    
    parameters: 
        X: features - np.array (N, # of features)
        clfs:  list of MyModel objects. each My Model object will need
        to have self.name, self. feat_idxs, self.feat_names, and self.model initialized
    
    returns:
        all_shap_values: a list of model name, explainer object, shapley values,
        and the associated feature names and indices

    '''
    
    all_shap_values = []
    #for label, model, _, _, feat_nums, feat_names, _, _, _, _, _ in all_num_feats:
    for clf in clfs:
        
        explainer = shap.KernelExplainer(clf.model.predict, X[:,clf.feat_idxs])
        shap_values = explainer.shap_values(X[:,clf.feat_idxs])
        
        all_shap_values.append([clf.name, explainer, shap_values, clf.feat_idxs, clf.feat_names])
        
        clf.shapley_values = [explainer, shap_values]
        

    return all_shap_values


def get_top_shap_names(clfs, val = 10):
    '''
    function: get_top_shap_names
    
    description: returns the names of the top #val Shapley values
        
    
    parameters: 
        clfs:  list of MyModel objects. each My Model object will need
        to have self.name, self. feat_idxs, self.feat_names, and self.model initialized
        val: optional, number of top Shapley names to return - int
    
    returns:
        top_shap: list of [model name, [top Shapley feature names]]



    '''
    top_shaps = []
    #for label, explainer, shap_values, feat_nums, feat_names in shapley:
    for clf in clfs:

        feature_names = clf.feat_names
        
        rf_resultX = pd.DataFrame(clf.shapley_values[1], columns = feature_names)
        
        vals = np.abs(rf_resultX.values).mean(0)
        
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                          columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                                       ascending=False, inplace=True)
        #print(shap_importance.head())
        
        top_shap = list(shap_importance['col_name'])[:val]
        
        top_shaps.append([clf.name, top_shap])
        
    return top_shaps