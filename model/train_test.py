# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:48:58 2023

@author: c_pia
"""
import logging
import sklearn
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

__all__ = ["five_fold_dataframe"]

logger = logging.getLogger(__name__)

def five_fold_dataframe(X_scaled, y, clfs):
    '''
    function: five_fold_dataframe
    
    description: complets five fold cross validation for all models indcluded in
    clfs
    
    parameters: 
        X scaled: model input, features - np.array (N, # of features)
        y: model output, class labels - 1D np.array (N,)
        clfs:  list of MyModel objects. each My Model object will need
        to have self.model initialized.
    
    returns:
        dfs: the respective metric score from each fold of a cross-validation
        - a dictonary of data frames. each data frame correspponds to metric score
        cm_scores: average, normalized confusion matrix per model - np.array
        mis_classes: all samples who that misclassifed -list of [model name, np.array]
    '''
    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=None)
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    kappa_scores = []
    mis_classes = []
    cm_scores = []
    col = []
    
    for clf in clfs: #all num feats is the object?
        
    #label, model, _, _, feat_nums, _
        acc_score = []
        prec_score = []
        rec_score = []
        f1_score = []
        kappa_score = []
        mis_class = []
        cm_score = []
        for train_index , test_index in kf.split(X_scaled,y):
            
            X_train , X_test = X_scaled[train_index,:],X_scaled[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
    
            clf.model.fit(X_train[:,clf.feat_idxs], y_train)
            pred_values = clf.model.predict(X_test[:,clf.feat_idxs])
             
            acc = sklearn.metrics.accuracy_score(y_test, pred_values)
            acc_score.append(acc)
            
            prec = sklearn.metrics.precision_score(y_test, pred_values)
            prec_score.append(prec)
            
            rec = sklearn.metrics.recall_score(y_test, pred_values)
            rec_score.append(rec)
            
            f1 = sklearn.metrics.f1_score(y_test, pred_values)
            f1_score.append(f1)
            
            kappa = sklearn.metrics.cohen_kappa_score(y_test, pred_values)
            kappa_score.append(kappa)
            
            mis_c_l = np.where(y_test != pred_values)[0]
            mis_c_g = test_index[mis_c_l]
            mis_class.append(mis_c_g)
            
            cm = sklearn.metrics.confusion_matrix(y_test, pred_values, normalize = 'true')
            cm_score.append(cm)
            
        mis_class = np.concatenate(mis_class)
        avg_cm_score = np.mean(np.array(cm_score), axis = 0)
        
        clf.mis_class = mis_class
        clf.performance = {'acc' : acc_score, 'prec' : prec_score, 'rec' : rec_score, 'f1' : f1_score, 'kappa' : kappa_score}

        
        acc_scores.append(acc_score)
        prec_scores.append(prec_score)
        rec_scores.append(rec_score)
        f1_scores.append(f1_score)
        kappa_scores.append(kappa_score)
        mis_classes.append([clf.name, mis_class])
        cm_scores.append([clf.name, avg_cm_score])
        col.append(clf.name)
        
    df_acc_score = pd.DataFrame(np.array(acc_scores).T, columns = col)
    df_prec_score = pd.DataFrame(np.array(prec_scores).T, columns = col)
    df_rec_score = pd.DataFrame(np.array(rec_scores).T, columns = col)
    df_f1_score = pd.DataFrame(np.array(f1_scores).T, columns = col)
    df_kappa_score = pd.DataFrame(np.array(kappa_scores).T, columns = col)
    df_f1_score = pd.DataFrame(np.array(f1_scores).T, columns = col)
    
    dfs = {'acc' : df_acc_score, 'prec' : df_prec_score, 'rec' : df_rec_score, 'f1' : df_f1_score, 'kappa' : df_kappa_score}
        

    return dfs, cm_scores, mis_classes
