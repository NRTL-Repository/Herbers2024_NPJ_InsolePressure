# -*- coding: utf-8 -*-

import logging
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

__all__ = ["p_val_filter"]

logger = logging.getLogger(__name__)


def p_val_filter(all_df, y):
    
    '''
    function: p_val_filter
    
    description: pre-filter features based on ANOVA and f-statistic. allfeatures
    with a p value less than 0.05 are removed
        
    
    parameters: 
        all_df: features - dataframe of size (n, # of features)
        y: groups - np.array (N,)

    
    returns:
        filtered_feat_locs: indices of features with p<0.05 - np.array
        filtered_feat_names: names of features with p<0.05 - list
        df_filtered_feats: filtered feature values - dataframe

    '''
    
    X = np.array(all_df)

    bestfeatures = SelectKBest(score_func=f_classif)
    fit = bestfeatures.fit(X, y)
    f_vals = np.expand_dims(fit.scores_, axis =0)
    p_vals = np.expand_dims(fit.pvalues_, axis =0)
    pf_vals = np.concatenate((p_vals, f_vals), axis = 0 )
    
    df_pf_vals = pd.DataFrame(data = pf_vals, columns = all_df.columns)
    
    thresholding = df_pf_vals.iloc[0] < 0.05
    filtered_feat_locs = np.where(thresholding == True)[0]
    filtered_feat_names = list(df_pf_vals.columns[filtered_feat_locs])
    
    df_filtered_feats = all_df[filtered_feat_names]

    return filtered_feat_locs, filtered_feat_names, df_filtered_feats

