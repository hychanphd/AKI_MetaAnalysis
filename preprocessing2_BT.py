'''
This module contains a set of function that process different table from preprocess1 module into one Big Table
Look at def bigtable

Input:
px_{site}_{str(year)}.pkl
dx_{site}_{str(year)}.pkl
onset_{site}_{str(year)}.pkl
vital_{site}_{str(year)}.pkls
amed_{site}_{str(year)}.pkl
lab_{site}_{str(year)}.pkl

Output:
bt3_{site}_{str(year)}.pkl
'''

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
import logging
import utils_function

def bt_onset(configs_variables, year):
    '''
    This module read and return the onset table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Merging onset on site '+site+":"+str(year), flush = True)    
    try:
        return pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
#        newdf_debug['onset'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No onset table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No onset table!!!!! '+site+":"+str(year))
        logging.shutdown()
        
def bt_px(configs_variables, year, newdf):        
    '''
    This module read the px table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    print('Merging px on site '+site+":"+str(year), flush = True)        
    try:
        px = pd.read_pickle(datafolder+site+'/px_'+site+'_'+str(year)+'.pkl')
        #depreciate Since ADMIT
#        newdf = pd.merge(newdf, px, left_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], right_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], how='left')
        return pd.merge(newdf, px, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna(False)
#        newdf_debug['px'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No px table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No px table!!!!! '+site+":"+str(year))
        logging.shutdown()
        return newdf
    
def bt_dx(configs_variables, year, newdf):                
    '''
    This module read the dx table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    
    print('Merging dx on site '+site+":"+str(year), flush = True)            
    try:
        dx = pd.read_pickle(datafolder+site+'/dx_'+site+'_'+str(year)+'.pkl')
#        newdf = pd.merge(newdf, dx, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')
        return pd.merge(newdf, dx, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna(False)       
#        newdf_debug['dx'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No onset table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No onset table!!!!! '+site+":"+str(year))
        logging.shutdown()
        return newdf
    
def bt_amed(configs_variables, year, newdf):                    
    '''
    This module read the amed table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Merging amed on site '+site+":"+str(year), flush = True)                
    try:
        amed = pd.read_pickle(datafolder+site+'/amed_'+site+'_'+str(year)+'.pkl')
#        newdf = pd.merge(newdf, amed, left_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], right_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], how='left')
        return pd.merge(newdf, amed, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna(False)        
#        newdf = newdf.combine_first(newdf[list(amed.select_dtypes('bool').columns)].fillna(False))    
#        newdf_debug['amed'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No amed table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No amed table!!!!! '+site+":"+str(year))
        logging.shutdown()    
        return newdf
    
def bt_labcat(configs_variables, year, newdf):                        
    '''
    This module read the lab(categorical) table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Merging lab_cat on site '+site+":"+str(year), flush = True)                
    try:
        labcat = pd.read_pickle(datafolder+site+'/labcat_'+site+'_'+str(year)+'.pkl')
#        newdf = pd.merge(newdf, amed, left_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], right_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], how='left')
        return pd.merge(newdf, labcat, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna(False)        
#        newdf = newdf.combine_first(newdf[list(amed.select_dtypes('bool').columns)].fillna(False))    
#        newdf_debug['amed'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No lab_cat table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No lab_cat table!!!!! '+site+":"+str(year))
        logging.shutdown()          
        return newdf
    
def bt_demo(configs_variables, year, newdf):                            
    '''
    This module read the demographic table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Merging demo on site '+site+":"+str(year), flush = True)                
    try:
        demo = pd.read_pickle(datafolder+site+'/demo_'+site+'_'+str(year)+'.pkl')
        newdf2 = pd.merge(newdf, demo, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')
        return newdf2.combine_first(newdf2[list(demo.select_dtypes('bool').columns)].fillna(False))            
#        newdf_debug['demo'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No demo table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No demo table!!!!! '+site+":"+str(year))
        logging.shutdown()    
        return newdf
    
def bt_vital(configs_variables, year, newdf):                                
    '''
    This module read the vital table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Merging vital on site '+site+":"+str(year), flush = True)                
    try:
#        vital = pd.read_pickle(datafolder+site+'/vital_'+site+'_'+str(year)+'.pkl')
        vital = pd.read_pickle(datafolder+site+'/vital_'+site+'_'+str(year)+'.pkl')

#        newdf = pd.merge(newdf, vital, left_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], right_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], how='left')
        return pd.merge(newdf, vital, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')        
#        newdf_debug['vital'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No vital table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No vital table!!!!! '+site+":"+str(year))
        logging.shutdown()
        return newdf
    
def bt_labnum(configs_variables, year, newdf):                                    
    '''
    This module read the lab(numerical) table and merge with the big table
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
        
    
    print('Merging lab_num on site '+site+":"+str(year), flush = True)                
    try:
#        labnum = pd.read_pickle(datafolder+site+'/labnum_'+site+'_'+str(year)+'.pkl')
        labnum = pd.read_pickle(datafolder+site+'/labnum_'+site+'_'+str(year)+'.pkl')

#        newdf = pd.merge(newdf, lab_t, left_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], right_on=['PATID', 'ENCOUNTERID', 'SINCE_ADMIT'], how='left')   
        return pd.merge(newdf, labnum, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')
#        newdf_debug['lab'] = newdf.copy()
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No lab_num table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No lab_num table!!!!! '+site+":"+str(year))
        logging.shutdown()
        return newdf
    
def drop_too_much_nan(configs_variables, year, newdf, threshold=0.05):
    '''
    This module read drop columns with 95% missing data in targeted class
    '''
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    
    print('Remove sparse feature on site '+site+":"+str(year), flush = True)                        
    btX = newdf.replace(False, np.nan)
    #limitPer = len(btX) * threshold
    #col = btX.dropna(thresh=limitPer, axis=1).columns
    btX0 = btX[btX['FLAG']==0]
    btX1 = btX[btX['FLAG']==1]
    limitPer0 = len(btX0) * threshold
    limitPer1 = len(btX1) * threshold
    col0 = btX0.dropna(thresh=limitPer0, axis=1).columns
    col1 = btX1.dropna(thresh=limitPer1, axis=1).columns
    col = list(set(list(col1)+list(col0)))
    return newdf[col]

def handpickremoval(configs_variables, year, newdf):   
    '''
    This module read drop the service CPT code and weight
    '''
    
    # drop CPT service code between 99202 and 99499 
    # cptcode0 = np.array([x for x in newdf.columns if 'PX' in x and x.split(':')[2].isnumeric()])
    # cptcode = np.array([int(x.split(':')[2]) if x.split(':')[2].isnumeric() else 0 for x in cptcode0])
    # cptcodebool = np.logical_or(np.logical_and(cptcode >= 99202, cptcode <= 99499),np.logical_and(cptcode >= 80047, cptcode <= 89398))
    # remlist = cptcode0[cptcodebool]       
    remlist = ['PX:CH:'+str(x) for x in range(99202,99500)]+['PX:CH:'+str(x) for x in range(80047,89399)]
    
    # Additional drop
    remlist2 = ['LAB::48642-3', 'LAB::48643-1']
    
    remlist = remlist+remlist2
    
    def check_columns_for_substrings(df, substrings):
        columns_with_substrings = [col for col in df.columns for substring in substrings if substring in col]
        return columns_with_substrings
    
    remlist3 = check_columns_for_substrings(newdf, remlist)
    
    # Additional drop weight
    remlist3 = remlist3+['WT']
    
    return newdf.drop(remlist3,axis=1, errors='ignore')

# def drop_corr(configs_variables, year, newdf, threshold=0.5):
#     print('Remove correlated feature on site '+site+":"+str(year), flush = True)                        
#     corr = newdf.corr()
#     columns = np.full((corr.shape[0],), True, dtype=bool)
#     for i in range(corr.shape[0]):
#         for j in range(i+1, corr.shape[0]):
#             if corr.iloc[i,j] >= threshold:
#                 # if corr.columns[j] == 'ORIGINAL_BMI':
#                 #     if columns[i]:
#                 #         columns[i] = False
#                 if columns[j]:
#                     columns[j] = False
#     selected_columns = newdf.columns[columns]
#     return newdf[selected_columns]

#Pearson
def pearson_list(bt, threshold):
    corr = bt.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= threshold:
#                print(bt.columns[i], btcont.columns[j], corr.iloc[i,j])
                if columns[j]:
                    columns[j] = False
    return columns   

def point_biserial(btcat, btcon, threshold):
    from scipy import stats
    columns = np.full((btcat.shape[1],), True, dtype=bool)    
    for i in range(btcon.shape[1]):
        for j in range(btcat.shape[1]):
            if stats.pointbiserialr(btcat.iloc[:,j], btcon.iloc[:,i])[0] >= threshold:            
                if columns[j]:
                    columns[j] = False
    return columns
    
def drop_corr(configs_variables, year, bt, threshold):
    bt2 = bt.reindex(sorted(bt.columns), axis=1)
    btcat = bt2.select_dtypes('bool')
    btcont = bt2.select_dtypes(exclude='bool')
    return pd.concat([btcont.loc[:,pearson_list(btcont, threshold)], btcat.loc[:, (pearson_list(btcat, threshold) | point_biserial(btcat, btcon, threshold))]], axis=1)

def drop_corr2(configs_variables, year, bt, threshold):
    return bt.loc[:,pearson_list(bt, threshold)]

def generate_drop_list(configs_variables, year, bt, threshold):
    corr = bt.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= threshold:
                print(configs_variables, year, bt.columns[i], btcont.columns[j], corr.iloc[i,j])        
                
def bigtable(configs_variables, year):
    '''
    This module combine different tables into one big table
    
    Input:
    px_{site}_{str(year)}.pkl
    dx_{site}_{str(year)}.pkl
    onset_{site}_{str(year)}.pkl
    vital_{site}_{str(year)}.pkl
    amed_{site}_{str(year)}.pkl
    lab_{site}_{str(year)}.pkl
    
    Output:
    bt3_{site}_{str(year)}.pkl - vital table (cont)    
    '''
    
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/bt3_'+site+'_'+str(year)+'.pkl'):
        print('Existed: bt3_'+site+'_'+str(year)+'.pkl')
        return     
    
    print('Merging bt on site '+site+":"+str(year), flush = True)

    #load tables
    try:
        #read onset table
        newdf = bt_onset(configs_variables, year)    
        # boolean table must merge first
        # merge boolean tables
        newdf = bt_px(configs_variables, year, newdf)
        newdf = bt_dx(configs_variables, year, newdf)
        newdf = bt_amed(configs_variables, year, newdf)
        newdf = bt_labcat(configs_variables, year, newdf)

        # merge continuous tables
        newdf = bt_demo(configs_variables, year, newdf)
        newdf = bt_vital(configs_variables, year, newdf)
        newdf = bt_labnum(configs_variables, year, newdf)
        newdf = handpickremoval(configs_variables, year, newdf)
        
        # Migrated to preprocessing4 module
#        newdf = drop_too_much_nan(configs_variables, year, newdf, threshold=0.05)
#        newdf = bt_postprocess(configs_variables, year, newdf)
#        newdf = drop_corr2(configs_variables, year, newdf, threshold=0.5)        
        
        #Save table
        newdf.to_pickle(datafolder+site+'/bt3_'+site+'_'+str(year)+'.pkl')

        #consistency check
        if newdf.empty:
            logging.basicConfig(filename='BT.log', filemode='a')    
            print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
            logging.error('BT: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
            logging.shutdown()

        print('Finished bt on site '+site+":"+str(year), flush = True)        
    except Exception as e:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('OTHER ERROR!!!!! '+site+":"+str(year)+'\n+++++++++++++++++\n'+str(e)+'\n-------------------\n', flush = True)
        logging.error('OTHER ERROR!!!!! '+site+":"+str(year)+'\n+++++++++++++++++\n'+str(e)+'\n-------------------\n')
        logging.shutdown()       
        raise    
        
def bigtable_removal_only(configs_variables, year, stg, newdf):
    '''
    Depreciated
    '''    
    #Big Table
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    print('BT removal on site '+site+":"+str(year), flush = True)

    try:
        newdf = pd.read_pickle(datafolder+site+'/bt3_'+site+'_'+stg+'_3000.pkl')      
        newdf = handpickremoval(configs_variables, year, newdf)        
        newdf.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+stg+'_3000.pkl')

        #consistency check
        if newdf.empty:
            logging.basicConfig(filename='BT.log', filemode='a')    
            print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
            logging.error('BT: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
            logging.shutdown()

        print('Finished bt on site '+site+":"+str(year), flush = True)        
    except Exception as e:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('OTHER ERROR!!!!! '+site+":"+str(year)+'\n+++++++++++++++++\n'+str(e)+'\n-------------------\n', flush = True)
        logging.error('OTHER ERROR!!!!! '+site+":"+str(year)+'\n+++++++++++++++++\n'+str(e)+'\n-------------------\n')
        logging.shutdown()       
        raise    
        
def bigtable_nocovid(configs_variables):
    
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/bt3nocovid_'+site+'_3000.pkl'):
        print('Existed: bt3nocovid_'+site+'_'+'3000.pkl')
        return     
    
    print('Processin bt3nocovid_ on site '+site+":", flush = True)
    
    
