# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:20:46 2023

@author: c_pia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_polar(df, y, features, name):
    df_y = pd.DataFrame(data = y, columns = ["group"])
    df_working = pd.concat([df_y, df[features]], axis = 1)
    df_good_feats = df_working
    df_good_feats_norm = (df_good_feats-df_good_feats.mean())/df_good_feats.std()
    t = df_good_feats_norm.groupby('group').mean()
    #u = t.diff().iloc[1]
    #u_ray = np.asarray(u)
    
    unique = np.unique(y)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    feats = len(features)
    step = 1/feats
    
    n0 = np.arange(0, 1, step) * 2*np.pi  
    n0 = np.concatenate((n0, n0[0:1]))
    
    n1 = np.arange(0, 1, step*1) * 2*np.pi  
    n1 = np.concatenate((n1, n1[0:1]))
    ax.set_xticks(n1)
    
    
    for i in range(len(unique)):
        ray = np.asarray(t.iloc[i])
        ray_plot = np.concatenate((ray,ray[0:1]))
        ax.plot(n0, ray_plot, linestyle='dashed', marker = '.', markersize=15, label = str(unique[i]))

    
    n2 = np.arange(0, 1, .01) * 2*np.pi  
    n2 = np.concatenate((n2, n2[0:1]))
    ax.plot(n2, np.zeros(n2.shape), color = 'black')
    ax.set_rlim(-1,1)
    ax.set_yticklabels([])
    
    
    d = np.asarray(np.arange(1, len(features)+1, 1))
    d = np.concatenate((d,d[0:1]))
    ax.set_xticklabels(d)
    #plt.savefig(name + '_polar.pdf') 
    
    
def plot_corr(df, features, name, x_axis_labels = None):
    
    if x_axis_labels is None:
        x_axis_labels = list(np.arange(0,len(features)))

    
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.color_palette("Spectral", as_cmap=True), vmin  =-1, vmax = 1,
                square=True, ax=ax,
                xticklabels=x_axis_labels, yticklabels=x_axis_labels,
                cbar_kws={'label': 'correlation'})
    
    #plt.savefig(name + '_corr.pdf') 
    
    
    
def plot_violin(df, y, features, name):
    
    df_y = pd.DataFrame(y, columns = ['group'])
    df_plt = pd.concat([df_y, df[features]], axis = 1)
    
    for j in range(0,len(features),12):
        c = 1
        plt.figure()
        feature_sub = features[j:j+12]
        for i in range(0, len(feature_sub)):
            ax = plt.subplot(4, 3,c)
            ax.set_box_aspect(1)
            ax = sns.violinplot(data = df_plt, x = 'group', y = feature_sub[i], split = False,
                           inner = 'quartile', medianprops=dict(color="white", alpha=0.7), showfliers = False, cut = 0)


            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(1)
                l.set_color('white')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(2)
                l.set_color('white')
                l.set_alpha(0.8)
                
            plt.ylabel('')
            plt.xlabel('group')
            ax.title.set_text(feature_sub[i])
            c = c+ 1
        #plt.savefig(name + str(j) + '_violin.pdf') 
        plt.show()
        
        
def plot_PCA(df, y, features, name):
    X = np.asarray(df[features])
    pca = PCA(n_components=2)
    pca.fit(X)
    t = pca.transform(X)
    
    group = y
    plt.figure()
    for i in np.unique(y):
        plt.plot(t[group==i,0], t[group==i,1], '.', markersize=20, label = str(i))
    
    #plt.plot(t[group==1,0], t[group==1,1], '.', color = 'blue', markersize=20)
    plt.legend()    
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    #plt.savefig(name + '_PCA.pdf') 



























