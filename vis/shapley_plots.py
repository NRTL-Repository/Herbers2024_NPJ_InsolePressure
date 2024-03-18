# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:51:13 2023

@author: c_pia
"""

import shap

import logging
import matplotlib.pyplot as plt

__all__ = ["plot_shapley"]

logger = logging.getLogger(__name__)

def plot_shapley(X, clfs, max_display = 10):
    
    
    for clf in clfs:

        plt.figure()
        plt.title(clf.name)
        shap.summary_plot(clf.shapley_values[1], X[:,clf.feat_idxs], feature_names = list(clf.feat_names), max_display = max_display)
        plt.show()

        plt.figure()
        plt.title(clf.name)
        shap.summary_plot(clf.shapley_values[1], X[:,clf.feat_idxs], feature_names = list(clf.feat_names), plot_type = 'bar', max_display = max_display)
        plt.show()

