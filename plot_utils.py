import importlib

import ipynb.fs.full.preprocessing0
import ipynb.fs.full.preprocessing05
#import ipynb.fs.full.prepossessing075_akistage
import preprocessing1
#import ipynb.fs.full.preprocessing2_BT
import preprocessing2_BT

import ipynb.fs.full.preprocessing25_BTcorr
import ipynb.fs.full.preprocessing3_smote
#import ipynb.fs.full.preprocessing4
import preprocessing4

#import ipynb.fs.full.runxgboost
import runxgboost

#import ipynb.fs.full.postprocessing1_SHAP

import postprocessing1_SHAP

import ipynb.fs.full.postprocessing3_collect

from ipynb.fs.full.slackbot import ping_slack
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
import csv
from dfply import *
from xgboost import XGBClassifier
import itertools
import os
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from os.path import exists
import logging
import time

import utils_function
import seaborn as sns
from scipy import stats
import pickle

def collect_cross_cv(configs_variables):
    folds = int(configs_variables[0]['n_splits'])
    shap_datas = list()
    for configs_variable_d in configs_variables:
        for configs_variable_m in configs_variables:
            for fold in range(folds):
                site_m, datafolder, home_directory = utils_function.get_commons(configs_variable_m)
                site_d, datafolder, home_directory = utils_function.get_commons(configs_variable_d)
                datafolder = configs_variable_m['datafolder']
                stg = configs_variable_m['stg']
                fs = configs_variable_m['fs']
                oversample = configs_variable_m['oversample']
                model_type = configs_variable_m['model_type']   
                drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']    
                year=3000
                if drop_correlation_catboost:
                    suffix = 'nc'
                else:
                    suffix = ''                    
                try:
                    shap_data = pd.read_pickle(datafolder+site_m+'/shapdata_cv_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl')
                    shap_datas.append(shap_data)
                except:
                    pass
    shap_datas = pd.concat(shap_datas).reset_index(drop=True)    
    return shap_datas

#Heatmap
def headmap(shap_datas, site_ano=False, target='roc'):
    shap_datas_agg = shap_datas[['site_m', 'site_d', target]].groupby(['site_m', 'site_d']).agg(['mean', 'std']).reset_index()

    site_key = {3:'MCW', 4:'UIOWA', 5:'UMHC', 6:'UNMC', 9:'UofU', 8:'UTHSCSA', 2:'KUMC', 1:'UTSW', 7:'UPITT'}
    site_keyr = {v: k for k, v in site_key.items()}
    
    if site_ano:
        shap_datas_agg['site_m'] = shap_datas_agg['site_m'].map(site_keyr)
        shap_datas_agg['site_d'] = shap_datas_agg['site_d'].map(site_keyr)

    shap_datas_agg.columns = ['site_m', 'site_d', 'mean', 'std']

    df = shap_datas_agg

    # Pivot the DataFrame to create a matrix for the heatmap
    pivot_mean = df.pivot("site_m", "site_d", "mean")
    pivot_std = df.pivot("site_m", "site_d", "std")

    # Plotting
    plt.figure(figsize=(10, 8.5))
    ax = sns.heatmap(pivot_mean, annot=False, fmt="", cmap='coolwarm')
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, left=False, bottom=False, top = False, labeltop=True)

    # Adding mean and standard deviation to the cell text
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            mean = pivot_mean.iloc[i, j]
            std = pivot_std.iloc[i, j]
            if pd.notna(mean) and pd.notna(std):
                text = f'{mean:.2f}\n({std:.3f})'
                plt.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontdict={'size': 10})

    # Rotate x-axis labels and move x-axis to the top
    ax.xaxis.set_label_position('top') 
    plt.xlabel('Source', fontsize=20)
    plt.ylabel('Validation', fontsize=20)
    
    if target=='roc':
        plt.savefig('Fig1_AUROC.svg', format='svg', bbox_inches='tight')
    else:
        plt.savefig('Fig1_PRAUC.svg', format='svg', bbox_inches='tight')
        

#Spider Plot
def create_radar_chart_with_confidence(shap_datas, site_ano=False, target='roc'):
    shap_datas_agg = shap_datas[['site_m', 'site_d', target]].groupby(['site_m', 'site_d']).agg(['mean', 'std']).reset_index()

    site_key = {3:'MCW', 4:'UIOWA', 5:'UMHC', 6:'UNMC', 10:'UofU', 8:'UTHSCSA', 2:'KUMC', 9:'UTSW', 7:'UPITT', 1:'IUR'}
    site_keyr = {v: k for k, v in site_key.items()}
    
    if site_ano:
        shap_datas_agg['site_m'] = shap_datas_agg['site_m'].map(site_keyr)
        shap_datas_agg['site_d'] = shap_datas_agg['site_d'].map(site_keyr)

    shap_datas_agg.columns = ['site_m', 'site_d', 'mean', 'std']

    df = shap_datas_agg    
    
    # Categories (site_d)
    categories = df['site_d'].unique()
    N = len(categories)

    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Draw each site_m with its confidence interval
    for site_m in df['site_m'].unique():
        site_df = df[df['site_m'] == site_m]
        values = site_df['mean'].tolist()
        values += values[:1]

        # Confidence interval calculation
        ci_upper = (site_df['mean'] + 1.96 * site_df['std']).tolist()
        ci_upper += ci_upper[:1]

        ci_lower = (site_df['mean'] - 1.96 * site_df['std']).tolist()
        ci_lower += ci_lower[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Model {site_m}')
        ax.fill_between(angles, ci_lower, ci_upper, alpha=0.4)

    # Add legend
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    if target=='roc':
        plt.savefig('AUROC_spider.svg', format='svg', bbox_inches='tight')
    else:
        plt.savefig('PRAUC_spider.svg', format='svg', bbox_inches='tight')
    
def get_importances_features_stat(configs_variables):
    #Feature rank map
    models = dict()

    sites = [configs_variable['site'] for configs_variable in configs_variables]
    configs_variable_m = configs_variables[0]
    datafolder = configs_variable_m['datafolder']
    stg = configs_variable_m['stg']
    fs = configs_variable_m['fs']
    oversample = configs_variable_m['oversample']
    model_type = configs_variable_m['model_type']   
    drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']    
    year=3000
    if drop_correlation_catboost:
        suffix = 'nc'
    else:
        suffix = ''      

    for site_m in sites:    
        for fold in range(10):
            models[(site_m, fold)] = pickle.load(open(datafolder+site_m+'/boosttrap_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl', 'rb'))

    df_importances = list()
    for site_m in sites:
        for fold in range(10):
            importances = models[(site_m, fold)][-3].get_feature_importance(prettified=True)
            importances['site'] = site_m
            importances['fold'] = fold
            importances['rank'] = importances['Importances'].rank(method='min', ascending=False)-1      

            # ##TEST
            # importances['Feature Id no unit'] = importances['Feature Id'].str.split('(').str[0]
            # importances = importances[importances['Feature Id no unit'] != 'LAB::LG49883-8']
            # importances = importances.drop('Feature Id no unit',axis=1)
            # ##TEST
            
            importances = importances[importances['rank']<100]
            importances['rank'] = (100-importances['rank'])/100
            df_importances.append(importances)

    df_importances = pd.concat(df_importances)
    df_importances = df_importances[['Feature Id', 'rank', 'site']].groupby(['Feature Id', 'site']).median().reset_index()
    df_importances['Feature Id no unit'] = df_importances['Feature Id'].str.split('(').str[0]

    ## TEST
    try:
        v19620 = df_importances[df_importances['Feature Id no unit']=='LAB::1962-0']['rank'].iloc[0]
        # The condition to check
        condition = (df_importances['Feature Id no unit'] == 'LAB::LG2807-8') & \
                    (df_importances['site'] == 'MCW') & \
                    (df_importances['Feature Id'] == 'LAB::LG2807-8(mmol/L)')

        # Update the rank value where the condition is True
        df_importances.loc[condition, 'rank'] = v19620
        df_importances = df_importances[df_importances['Feature Id no unit']!='LAB::1962-0']
    except:
        pass
    
    df_importances = df_importances.sort_values('rank', ascending=False).groupby(['Feature Id no unit', 'site']).first().reset_index()

    df_importances_raw2 = df_importances.copy()

    df_importances_stat = df_importances[['Feature Id no unit', 'rank']].groupby(['Feature Id no unit']).quantile([0.25, 0.5, 0.75]).reset_index().pivot(index='level_1', columns=['Feature Id no unit'], values='rank').T.reset_index()
    df_importances_stat['IQR'] = df_importances_stat[0.75]-df_importances_stat[0.25]
    df_importances_stat.index = df_importances_stat['Feature Id no unit'].copy()
    df_importances_stat = df_importances_stat[[0.5, 'IQR']]
    df_importances_stat.columns = ['Median', 'IQR']

    df_importances_count = df_importances[['Feature Id no unit', 'site']].groupby('Feature Id no unit').count()
    df_importances_count.columns = ['Count']

    df_importances_stat = df_importances_stat.merge(df_importances_count, left_index=True, right_index=True)

    df_importances_top5 = df_importances_stat[['Median', 'Count']].sort_values('Median',ascending=False).groupby('Count').rank(method='first', ascending=False)
    df_importances_top5.columns = ['Label_rank']

    df_importances_top5=df_importances_top5[df_importances_top5['Label_rank']<=5]
    df_importances_top5['Label_rank'] = -1*(df_importances_top5['Label_rank']-3)-2
    df_importances_stat = df_importances_stat.merge(df_importances_top5[['Label_rank']], left_index=True, right_index=True, how='left').fillna(-100)
    
    return df_importances, df_importances_stat


def plot_feature_importance(df_importances_stat2, translator, modify_list, include_list):
    
    df_importances_stat = df_importances_stat2.copy().drop(['Label_rank'],axis=1)
    df_importances_top5 = df_importances_stat[['Median', 'Count']].sort_values('Median',ascending=False).groupby('Count').rank(method='first', ascending=False)
    df_importances_top5.columns = ['Label_rank']

    df_importances_top5=df_importances_top5[(df_importances_top5['Label_rank']<=5) | df_importances_top5.index.str.contains('|'.join(include_list))]
    df_importances_top5['Label_rank'] = -1*(df_importances_top5['Label_rank']-3)-2
    df_importances_stat = df_importances_stat.merge(df_importances_top5[['Label_rank']], left_index=True, right_index=True, how='left').fillna(-100)   
    
    #Customize height if overlap
    df_importances_stat = df_importances_stat2.copy()
    df_importances_stat['cusheight'] = 1
    for mf in modify_list:
        df_importances_stat.loc[mf, 'cusheight'] = 1.75

    df = df_importances_stat
    plt.figure(figsize=(20, 12))
    
    scatter = plt.scatter(df['Median'], df['Count'], c=df['IQR'], cmap='coolwarm', s=200)

    # Colorbar for IQR
    cbar  = plt.colorbar(scatter, label='IQR Value')
    cbar.ax.tick_params(labelsize=12)  # Increase font size for color bar ticks
    cbar.set_label('IQR Value', fontsize=20)  # Increase font size for color bar label

    # Add text for points where Label is True with annotation linesdd
    for idx, row in df.iterrows():
        if row['Label_rank']!=-100:
            plt.annotate(
                translator(idx), 
                xy=(row['Median'], row['Count']), 
                xytext=(row['Median'] + 0.07*row['Label_rank']-0.185, row['Count'] + 0.3*[-row['cusheight'] if row['Label_rank']%2==1 else row['cusheight']][0]),  # adjust text position
                arrowprops=dict(arrowstyle='-', lw=1),
                fontsize=9
            )

    # Set y-ticks to be at intervals of 0.1
    plt.yticks(np.arange(0, 10, 1))
    plt.ylim([0, 9.7])
    
    # Adding grid lines
    plt.grid(True)

    # Increase font size for x and y ticksd
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    

    plt.xlabel('Importance ranking (median of soft ranking)', fontsize = 20)
    plt.ylabel('Commonality across sites', fontsize= 20)
    plt.savefig('feature_ranks.svg', bbox_inches='tight')
    
def geb_top_top(configs_variables, top0=10):
    
    # get top features
    df_importances, df_importances_stat = get_importances_features_stat(configs_variables)
    df = df_importances.sort_values('rank', ascending=False).reset_index().groupby('site').head(top0)
    top3030 = df[['site', 'Feature Id']].groupby('Feature Id').count().sort_values('site',ascending=False).head(top0)  
    top3030.to_parquet("toptop.parquet")
    
    
def plot_feature_importancer2(df_importances_stat2, external_heatmap_df, translator, modify_list, include_list):

    df_importances_stat = df_importances_stat2.copy().drop(['Label_rank'],axis=1)
    df_importances_top5 = df_importances_stat[['Median', 'Count']].sort_values('Median',ascending=False).groupby('Count').rank(method='first', ascending=False)
    df_importances_top5.columns = ['Label_rank']

    df_importances_top5=df_importances_top5[(df_importances_top5['Label_rank']<=5) | df_importances_top5.index.str.contains('|'.join(include_list))]
    df_importances_top5['Label_rank'] = -1*(df_importances_top5['Label_rank']-3)-2
    df_importances_stat = df_importances_stat.merge(df_importances_top5[['Label_rank']], left_index=True, right_index=True, how='left').fillna(-100)    
    
    #Customize height if overlap
    
    df_importances_stat['cusheight'] = 1
    for mf in modify_list:
        df_importances_stat.loc[mf, 'cusheight'] = 1.75
    
    external_heatmap_df = external_heatmap_df.rename(columns={'0':'Feature Id', 'r.sq_spline_noAUC':'heap'})
    external_heatmap_df['Feature Id no unit'] = external_heatmap_df['Feature Id'].str.split('(').str[0]
    external_heatmap_df.index = external_heatmap_df['Feature Id no unit']    
    
    plt.figure(figsize=(20, 12))
    
    df_importances_stat = df_importances_stat.merge(external_heatmap_df, left_index=True, right_index=True, how='inner')
    df = df_importances_stat
    print(df.columns)
    scatter = plt.scatter(df['Median'], df['Count'], c=df['heap'], cmap='coolwarm', s=200)

    # Colorbar for IQR
    cbar  = plt.colorbar(scatter, label=f'$r^2$')
    cbar.ax.tick_params(labelsize=12)  # Increase font size for color bar ticks
    cbar.set_label(f'$r^2$', fontsize=20)  # Increase font size for color bar label

    # Add text for points where Label is True with annotation linesdd
    for idx, row in df.iterrows():
        if row['Label_rank']!=-100:
            plt.annotate(
                translator(idx), 
                xy=(row['Median'], row['Count']), 
                xytext=(row['Median'] + 0.007*row['Label_rank']-0.055, row['Count'] + 0.3*[-row['cusheight'] if row['Label_rank']%2==1 else row['cusheight']][0]),  # adjust text position
                arrowprops=dict(arrowstyle='-', lw=1),
                fontsize=9
            )

    # Set y-ticks to be at intervals of 0.1
    plt.yticks(np.arange(0, 10, 1))
    plt.ylim([5, 9.7])
    
    # Adding grid lines
    plt.grid(True)

    # Increase font size for x and y ticksd
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
    
    plt.xlabel('Importance ranking (median of soft ranking)', fontsize = 20)
    plt.ylabel('Commonality across sites', fontsize= 20)
    plt.savefig('feature_ranksr2.svg', bbox_inches='tight')