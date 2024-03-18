# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:48:36 2023

@author: c_pia
"""
#%% Imports
import pandas as pd
import numpy as np
from functions import filters, preprocess, wrappers
from vis import model_plots, feature_plots, shapley_plots
from model import train_test, shapley
from model.build_myModel import MyModel
from common import features, misclassified
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import matplotlib.pyplot as plt
    
#%% Load in data
df_avg_s = pd.read_csv('./data/df_avg_s.csv')
df_avg_a = pd.read_csv('./data/df_avg_a.csv') 
df_asym_s = pd.read_csv('./data/df_asym_s.csv')
df_asym_a = pd.read_csv('./data/df_asym_a.csv')

df_y = pd.read_csv('./data/df_y.csv')

#BINARY LABEL controls vs. PD
#0 is control, 1 is PD
y_group = np.array(df_y)[:,0]

#BINARY LABEL nonfaller vs. faller 
#0 is nonfaller, 1 is faller
y_falls = np.array(df_y)[:,1]

#MULTIVARIATE LABEL control vs. PD nonfaller vs. PD faller 
#0 is control, 1 is PD nonfaller, 2 is PD faller
y_multi = y_group + y_falls

#BINARY LABEL PD nonfaller vs. PD faller 
#0 is PD nonfaller, 1 is PD faller
y_PDonly_falls = y_falls[y_group == 1]

y_age = np.load('./data/y_age.npy')
match = np.where((y_age >38))[0]

one = y_group == 1
two = y_age<=38

op_match = np.where((one | two ))[0]



#%% create data frames for static, active, static + active
#as_df = pd.concat([df_avg_s[df_avg_s.columns[1:]], df_avg_a[df_avg_a.columns[1:]]], axis = 1)
#a_df = df_avg_a[df_avg_a.columns[1:]]
#s_df = df_avg_s[df_avg_s.columns[1:]]


as_df = pd.concat([df_avg_s[df_avg_s.columns[1:]], df_avg_a[df_avg_a.columns[1:]],\
                    df_asym_s[df_asym_s.columns[1:]], df_asym_a[df_asym_a.columns[1:]]], axis = 1)
    
a_df = pd.concat([df_avg_a[df_avg_a.columns[1:]], \
                    df_asym_a[df_asym_a.columns[1:]]], axis = 1)
    
s_df = pd.concat([df_avg_s[df_avg_s.columns[1:]], \
                    df_asym_s[df_asym_s.columns[1:]]], axis = 1)

balance_tasks = 'Static+Active'
isolate_PD = True
age_match = False
op_age_match = False


if balance_tasks == 'Static':
    df_working = s_df
elif balance_tasks == 'Active':
    df_working = a_df
elif balance_tasks == 'Static+Active':
    df_working = as_df

print('Assessing ' + balance_tasks)

if isolate_PD:
    assessment = 'PDnf_PDf'
    df_working = df_working[y_group == 1].reset_index(drop = True)
    y_working = y_PDonly_falls #assing PD nonfallers PD fallers
    print('Classifying PD nonfaller vs. PD fallers')
    folds = 3
    scoring = 'f1'
elif age_match:
    assessment = 'age_match'
    print('Classifying age match control vs. PD')
    df_working = df_working.loc[match].reset_index(drop = True)
    y_working = y_group[match]
    folds = 5
    scoring = 'f1'
elif op_age_match:
    assessment = 'young'
    print('Classifying young control vs. PD')
    df_working = df_working.loc[op_match].reset_index(drop = True)
    y_working = y_group[op_match]
    folds = 5
    scoring = 'f1'
else:
    assessment = 'all'
    print('Classifying control vs. PD')
    y_working = y_group 
    folds = 5
    scoring = 'f1'
    
    #can use code below for multivariate classifier
    #currently commented out
    #print('Classifying control vs. PD nonfaller vs. PD faller')
    #y_working = y_multi 
    #scoring = 'f1_weighted'

name = balance_tasks + assessment
print(name)
#%% pre-processing
#p value filter
filtered_feat_locs, filtered_feat_names, df_filtered_feats = filters.p_val_filter(df_working, y_working)

#scale data
X_scaled = preprocess.scale_data(df_filtered_feats)
#%%forward sequential feature selection.

clf_svm = SVC(kernel = 'linear')
clf_rf = RandomForestClassifier(max_depth=5, random_state=0)
clf_lr = LogisticRegression(random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_gnb = GaussianNB()

#%%
mySVM = MyModel(str(clf_svm), clf_svm)
myRF = MyModel(str(clf_rf), clf_rf)
myLR = MyModel(str(clf_lr), clf_lr)
myKNN = MyModel(str(clf_knn), clf_knn)
myGNB = MyModel(str(clf_gnb), clf_gnb)

myCLFs = [mySVM, myRF, myLR, myKNN, myGNB]
#%%
for clf in myCLFs:
    sfs_out = wrappers.fsfs(clf.model, X_scaled, y_working, scoring = scoring, folds = folds)
    clf.define_summary(sfs_out)
    clf.calculate_optimized(df_working[filtered_feat_names])

#%% summary of model
model_plots.plot_f1_summary(myCLFs, name )

#%% five fold model performance
dfs, cm_scores, mis_classes = train_test.five_fold_dataframe(X_scaled, y_working, myCLFs)
model_plots.plot_performnce_scores(dfs, ylim = [0.5,1.05], name = name)

#%% roc plots
model_plots.roc_plots(X_scaled, y_working, myCLFs, balance_tasks)
#%% find common features
common_features_idx, common_features = features.find_common_features(myCLFs, filtered_feat_names)
common_mis_idx, common_mis_count = misclassified.find_common_misclassified(myCLFs, y_working)
common_features_idx, common_features = features.find_common_features(myCLFs, filtered_feat_names)
common_mis_idx, common_mis_count = misclassified.find_common_misclassified(myCLFs, y_working)

#%% shapley values
all_shapley_values = shapley.shapley(X_scaled, myCLFs) #takes long 
top_shapley_names = shapley.get_top_shap_names(myCLFs, val = 1)

#%% feature intersection
feature_intersection = features.find_feature_intersection(common_features, top_shapley_names)
feature_plots.plot_violin(df_working, y_working, feature_intersection, name + 'feat_intersection')

#%% feature plots

feature_plots.plot_polar(df_working, y_working, common_features, name)
feature_plots.plot_corr(df_working, common_features, name)
feature_plots.plot_violin(df_working, y_working, common_features, name)
feature_plots.plot_PCA(df_working, y_working, common_features, name)

#%% shapley plots
shapley_plots.plot_shapley(X_scaled, myCLFs, max_display = 5)

