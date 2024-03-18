# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:00:33 2023

@author: c_pia
"""
import logging
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy.stats import sem
import math

__all__ = ["plot_f1_summary", "plot_performnce_scores", "roc_plots"]

logger = logging.getLogger(__name__)

def plot_f1_summary(clfs, title):
    '''
    
    function: plot_f1_summary
    
    description: plot the f1 score from each iteration of forward sequential
    feature selection
    
    parameters: 
        clfs: a list of MyModel objects. each My Model object will need
        to have self.model_fsfs initialized.
        title: desired title of plot - string
    
    returns: N/A
    
    '''
    plt.figure()
    plt.title(title)
    for clf in clfs:
        clf.model_fsfs['avg_score'].plot(label = clf.name)
    plt.legend()


def plot_performnce_scores(dfs, ylim = [0.5,1.0], name = ""):
    '''
    function: plot_performnce_scores
    
    description: plot the accuract, precision, recall, f1, and kappa scores of 
    each model
    
    parameters: 
        dfs: the respective metric score from each fold of a cross-validation
        - a dictonary of data fromes. each data frame correspponds to metric score
        ylim: optional, desired ylim of metrics plots - range
        name: optional, desired title of plot - string
    
    returns: N/A    
    '''
    plt.figure()
    
    #cols = ['svm', 'rf', 'lr', 'knn', 'gnb'] #fix cols

    fig, ax = plt.subplots(5,1)
    
    df_acc_scores = dfs['acc']
    means = df_acc_scores.mean()
    error = df_acc_scores.sem()
    means.plot.bar(yerr = error, ax = ax[0], capsize=4, rot=0, width=1)
    ax[0].set_ylim(ylim)
    
    df_prec_scores = dfs['prec']
    means = df_prec_scores.mean()
    error = df_prec_scores.sem()
    means.plot.bar(yerr = error, ax = ax[1], capsize=4, rot=0, width=1)
    ax[1].set_ylim(ylim)
    
    df_rec_scores = dfs['rec']
    means = df_rec_scores.mean()
    error = df_rec_scores.sem()
    means.plot.bar(yerr = error, ax = ax[2], capsize=4, rot=0, width=1)
    ax[2].set_ylim(ylim)
    
    df_f1_scores = dfs['f1']
    means = df_f1_scores.mean()
    error = df_f1_scores.sem()
    means.plot.bar(yerr = error, ax = ax[3], capsize=4, rot=0, width=1)
    ax[3].set_ylim(ylim)
    
    df_kappa_scores = dfs['kappa']
    means = df_kappa_scores.mean()
    error = df_kappa_scores.sem()
    means.plot.bar(yerr = error, ax = ax[4], capsize=4, rot=0, width=1)
    ax[4].set_ylim(ylim)
    
    fig.set_size_inches(3,9)
    #plt.savefig( name +'_performance.pdf') #just active tasks to save time
    plt.show()
    
def roc_plots(X_scaled, y, clfs, assessment):
    '''
    function: roc_plots
    
    description: plot ROC curve from each fold of a cross-validation, as well
    as the average ROC curve
    
    parameters: 
        X scaled: model input, features - np.array (N, # of features)
        y: model output, class labels - 1D np.array (N,)
        clfs:  list of MyModel objects. each My Model object will need
        to have self.model initialized.
    
    returns: N/A    
    
    '''
    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=None)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tprs = []
    c = 0
    mods = ['SVM', 'RF', 'LR', 'KNN', 'GNB']

    #for label, model, _, _, feat_nums, _ in all_num_feats:
        
    #fig, axs = plt.subplots(3, 2, figsize=(8,10))
    
    plt.subplots(figsize=(14, 16))
    plt.subplots_adjust(wspace=0)
    bar_dict = {}
    for c, clf in enumerate(clfs):
        
        
        tprs = []
        aucs = []
        #fig, ax = plt.subplots(3,2,figsize=(6, 6))
        
        ax = plt.subplot(3,2,c+1)
        

        for fold, (train_index , test_index) in enumerate(kf.split(X_scaled,y)):
            
            X_train , X_test = X_scaled[train_index,:],X_scaled[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
    
            clf.model.fit(X_train[:,clf.feat_idxs], y_train)

            dis = sklearn.metrics.RocCurveDisplay.from_estimator(clf.model, \
                                                             X_test[:,clf.feat_idxs], \
                                                            y_test,\
                                                                drop_intermediate = False,\
                                                                    name=f"ROC fold {fold}",\
                                                                  ax = ax)

            interp_tpr = np.interp(mean_fpr, dis.fpr, dis.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            aucs.append(dis.roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_tprs.append(mean_tpr)
        
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        sem_auc = sem(aucs)
        bar_dict[clf.name] = [mean_auc, sem_auc]
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=mods[c]
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        #plt.title(mods[c])
        c=c+1
    plt.suptitle(assessment + ':\nROC curves for PD non-fallers vs. PD fallers classifiers')
    path = r'C:\Users\c_pia\Working Drive\Research\Thesis\UMN-PhD-Thesis-Template-master - Copy\UMN-Thesis-New\figures\appendix'
    #plt.savefig( path +'\\'+assessment +'_roc.pdf') #just active tasks to save time
    plt.show()
    plt.figure()
    colors = ['tab:purple', 'tab:orange','tab:red','tab:green','tab:blue']
    labels = ['SVM', 'RF', 'LR', 'KNN', 'GNB']
    for i, (k,v) in enumerate(bar_dict.items()):
        plt.bar(i, v[0], yerr = v[1], color = colors[i])
        print(k)
        print(v[1])
        plt.text(i - 0.25, 0.8, '%.2f' % v[0] , fontsize=15, color = 'white')
    
    plt.xticks(np.arange(0,5), labels)
    plt.ylim([0.7, 1])
        
    
    
    
        

