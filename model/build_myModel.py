# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:00:24 2023

@author: c_pia
"""

import numpy as np

class MyModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def define_summary(self, model_fsfs):
        self.model_fsfs = model_fsfs

    def define_optimized(self, num_feats, f1, feat_idxs, feat_names, mis_class = None, performance = None, shapley_values = None):
        self.num_feats = num_feats
        self.f1 = f1
        self.feat_idxs = feat_idxs
        self.feat_names = feat_names
        self.mis_class = mis_class
        self.performance = performance
        self.shapley_values = shapley_values
        
        
    def calculate_optimized(self, all_df, model_fsfs = None):
        
        if model_fsfs is not None:
            self.model_fsfs = model_fsfs
            print('not none')
            
        f1 = np.asarray(self.model_fsfs['avg_score'])
        ind_f1 = np.argsort(f1)[-1:]
        iloc_good = self.model_fsfs.iloc[ind_f1[-1]]
        
        avg_score = iloc_good['avg_score']
        num_feats = len(iloc_good['feature_idx'])
        
        feat_nums = [int(numeric_string) for numeric_string in iloc_good['feature_names']]
        feat_names = all_df.columns[feat_nums]
        
        
        self.num_feats = num_feats
        self.f1 = avg_score
        self.feat_idxs = feat_nums
        self.feat_names = feat_names
        