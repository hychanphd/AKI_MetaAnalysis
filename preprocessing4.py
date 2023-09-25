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

from joblib import Parallel, delayed
from joblib import parallel_backend

def drop_too_much_nan(site, year, newdfs, threshold, keep_med=True):
    print('Remove sparse feature on site '+site+":"+str(year), flush = True)                        
    allcols = []
    for newdf in newdfs:
        allcols = allcols + list(newdf.columns)
    allcols = np.unique(np.array(allcols))
    allcols = allcols[allcols != 'FLAG']
    allcols = allcols[allcols != 'PATID']
    allcols = allcols[allcols != 'ENCOUNTERID']

    flag0nan = {key: 0 for key in allcols}
    flag1nan = {key: 0 for key in allcols}
    flag0total = 0
    flag1total = 0

    for newdf in newdfs:
        btX = newdf.replace(False, np.nan)
        flag0total += np.logical_not(btX['FLAG']).sum()
        flag1total += btX['FLAG'].sum()    
        for col in allcols:
            if col in newdf.columns:
                flag0nan[col] += np.logical_and(np.logical_not(btX['FLAG']), np.isnan(btX[col])).sum()
                flag1nan[col] += np.logical_and(btX['FLAG'], np.isnan(btX[col])).sum()
            else:
                flag0nan[col] += np.logical_not(btX['FLAG']).sum()
                flag1nan[col] += btX['FLAG'].sum()
                
    remlist = []        
    for col in allcols:
#        print(col, flag0nan[col]/flag0total, flag1nan[col]/flag1total)        
        if flag0nan[col]/flag0total >= 1-threshold and flag1nan[col]/flag1total >= 1-threshold:
            remlist = remlist + [col]

    if keep_med:
        remlist = [x for x in remlist if 'MED' not in x]
            
    for i in range(len(newdfs)):
        newdfs[i] = newdfs[i].drop(remlist,axis=1, errors='ignore')

    return newdfs, remlist, flag0nan, flag1nan, flag0total, flag1total



def drop_too_much_nan_positive(site, year, newdfs, threshold, keep_med=True):
    print('Remove sparse feature on site '+site+":"+str(year), flush = True)                        
    allcols = []
    for newdf in newdfs:
        allcols = allcols + list(newdf.columns)
    allcols = np.unique(np.array(allcols))
    allcols = allcols[allcols != 'FLAG']
    allcols = allcols[allcols != 'PATID']
    allcols = allcols[allcols != 'ENCOUNTERID']

    flag0nan = {key: 0 for key in allcols}
    flag1nan = {key: 0 for key in allcols}
    flag0total = 0
    flag1total = 0

    for newdf in newdfs:
        btX = newdf.replace(False, np.nan)
        flag0total += np.logical_not(btX['FLAG']).sum()
        flag1total += btX['FLAG'].sum()    
        for col in allcols:
            if col in newdf.columns:
#                flag0nan[col] += np.logical_and(np.logical_not(btX['FLAG']), np.isnan(btX[col])).sum()
                flag1nan[col] += np.logical_and(btX['FLAG'], np.isnan(btX[col])).sum()
            else:
#                flag0nan[col] += np.logical_not(btX['FLAG']).sum()
                flag1nan[col] += btX['FLAG'].sum()

    remlist = []        
    for col in allcols:
#        print(col, flag0nan[col]/flag0total, flag1nan[col]/flag1total)        
#        if flag0nan[col]/flag0total >= 1-threshold and flag1nan[col]/flag1total >= 1-threshold:
        if flag1nan[col]/flag1total >= 1-threshold:
            remlist = remlist + [col]
            
    if keep_med:
        remlist = [x for x in remlist if 'MED' not in x]
        
    for i in range(len(newdfs)):
        newdfs[i] = newdfs[i].drop(remlist,axis=1, errors='ignore')
        
    return newdfs, remlist, flag0nan, flag1nan, flag0total, flag1total

def drop_too_much_nan_positive_parallel(site, year, newdfs, threshold, keep_med=True, n_jobs=1):
    print('Remove sparse feature on site '+site+":"+str(year), flush = True)                        
    allcols = []
    for newdf in newdfs:
        allcols = allcols + list(newdf.columns)
    allcols = np.unique(np.array(allcols))
    allcols = allcols[allcols != 'FLAG']
    allcols = allcols[allcols != 'PATID']
    allcols = allcols[allcols != 'ENCOUNTERID']

    def nan_count(newdf, allcols):
        flag0nan = {key: 0 for key in allcols}
        flag1nan = {key: 0 for key in allcols}
        flag0total = 0
        flag1total = 0

        btX = newdf.replace(False, np.nan)
        flag0total += np.logical_not(btX['FLAG']).sum()
        flag1total += btX['FLAG'].sum()    
        for col in allcols:
            if col in newdf.columns:
                flag1nan[col] += np.logical_and(btX['FLAG'], np.isnan(btX[col])).sum()
            else:
                flag1nan[col] += btX['FLAG'].sum()
        return flag0nan, flag1nan, flag0total, flag1total

    nan_list = Parallel(n_jobs=n_jobs)(delayed(nan_count)(newdf, allcols) for newdf in newdfs)

    flag0nan = {key: 0 for key in allcols}
    flag1nan = {key: 0 for key in allcols}
    flag0total = 0
    flag1total = 0

    for i in range(len(nan_list)):
        for k, v in nan_list[i][0].items():
            flag0nan[k] += v
        for k, v in nan_list[i][1].items():
            flag1nan[k] += v
        flag0total += nan_list[i][2]
        flag1total += nan_list[i][3]        
        
    remlist = []        
    for col in allcols:
#        print(col, flag0nan[col]/flag0total, flag1nan[col]/flag1total)        
#        if flag0nan[col]/flag0total >= 1-threshold and flag1nan[col]/flag1total >= 1-threshold:
        if flag1nan[col]/flag1total >= 1-threshold:
            remlist = remlist + [col]
            
    if keep_med:
        remlist = [x for x in remlist if 'MED' not in x]
        
    for i in range(len(newdfs)):
        newdfs[i] = newdfs[i].drop(remlist,axis=1, errors='ignore')
        
    return newdfs


def bt_ckd(site, year, newdf):                                    
    #lab_num
    print('Merging ckd_info on site '+site+":"+str(year), flush = True)                
    try:
        efgr2 = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_'+'ckdgroup'+'_'+site+'.pkl')
        return pd.merge(newdf, efgr2, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')
    except FileNotFoundError:
        logging.basicConfig(filename='BT.log', filemode='a')    
        print('No efgr table!!!!! '+site+":"+str(year), flush = True)
        logging.error('No efgr table!!!!! '+site+":"+str(year))
        logging.shutdown()
        return newdf

def bt_postprocess(site, year, newdf):
    print('Finishing on site '+site+":"+str(year), flush = True)                    
    newdf = newdf.drop(['PATID', 'ENCOUNTERID', 'AKI1_SINCE_ADMIT', 'SINCE_ADMIT', 'DAYS_SINCE_ADMIT','DAYS_SINCE_ADMIT_x'],axis=1, errors='ignore')
    newdf.columns=newdf.columns.str.replace('<','st')
    newdf.columns=newdf.columns.str.replace('>','bt')
    newdf.columns=newdf.columns.str.replace('[','lb')
    newdf.columns=newdf.columns.str.replace(']','rb')   
    return newdf.dropna(axis=1, how='all')

#    newdf_debug['drop'] = newdf.copy()



def flag_convert(dataX, stg):
    data = dataX.copy()
    
    if stg == 'stg23':
        data = data[data['FLAG']!=1]
        data['FLAG'] = (data['FLAG']>1)*1
        return data
    
    if stg == 'stg010':
        data = data[data['FLAG']!=2]
        data = data[data['FLAG']!=3]
        return data
    
    if stg == 'stg123':
        data = data[data['FLAG']!=0]
        
    if stg == 'stg01':
        data['FLAG'] = (data['FLAG']>0)*1
    else:
        data['FLAG'] = (data['FLAG']>1)*1    

    return data



def combinebt(site, yearX, stg, threshold=0.01):
    
    onset = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_onset_'+site+'.pkl')
    years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
    bt_list = list()

    for year in years:
        try:
            data = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+str(year)+'.pkl')
            data = flag_convert(data, stg)
            bt_list.append(data.copy())
        except:
            print(str(year)+' not exists')
            
    bt_list, remlist, flag0nan, flag1nan, flag0total, flag1total = drop_too_much_nan(site, yearX, bt_list, threshold)
    bt_all = pd.concat(bt_list, ignore_index=True)
    # replace nan in boolean columns with False
    bt_bool = bt_all.select_dtypes('O').columns
    bt_all[bt_bool] = bt_all[bt_bool].fillna(False)

    bt_all = bt_ckd(site, yearX, bt_all)
    bt_all = bt_postprocess(site, yearX, bt_all)
    bt_all.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+stg+'_3000.pkl')

def correct_dtypes(datanew, site, stg): 
#    datanew = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3pos_'+site+'_'+stg+'_3000.pkl')
    
    onset = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_onset_'+site+'.pkl')
    years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique()) 
    bt_list = list()

    for year in years:
        try:
            data = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+str(year)+'.pkl')
            bt_list.append(data.dtypes)
        except:
            print(str(year)+' not exists')

    datatype_list = pd.concat(bt_list).reset_index().drop_duplicates()
    datatype_list['index'] = datatype_list['index'].str.replace('<','st')
    datatype_list['index'] = datatype_list['index'].str.replace('>','bt')
    datatype_list['index'] = datatype_list['index'].str.replace('[','lb')
    datatype_list['index'] = datatype_list['index'].str.replace(']','rb') 
    datatype_list.index = datatype_list['index']
    datatype_list.columns = ['index1', 0]

    datanewcol = pd.DataFrame(datanew.columns)
    datanewcol.columns = ['index1']

    datatype_list = datanewcol.merge(datatype_list, on='index1', how='left')
    datatype_list.index = datatype_list['index1']
    datatype_list = datatype_list[0].to_dict()
    datanew = datanew.astype(datatype_list)
    return datanew    
    
def combinebtpos_old(site, yearX, stg, threshold=0.01, n_jobs=1):

    onset = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_onset_'+site+'.pkl')
    years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
    bt_list = list()

    for year in years:
        try:
            data = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+str(year)+'.pkl')
            data = flag_convert(data, stg)
            # col = data.columns
            # n=int(data.shape[0]/n_jobs)+1
            # chunks = [data[i:i+n].copy() for i in range(0,data.shape[0],n)]            
            # bt_list = bt_list+chunks
            bt_list.append(data.copy())            
        except:
            print(str(year)+' not exists')

    #    bt_list, remlist, flag0nan, flag1nan, flag0total, flag1total = drop_too_much_nan(site, yearX, bt_list, threshold)
    bt_list, remlist, flag0nan, flag1nan, flag0total, flag1total = drop_too_much_nan_positive(site, yearX, bt_list, threshold)
#    bt_list = drop_too_much_nan_positive_parallel(site, yearX, bt_list, threshold, n_jobs=n_jobs)

#    xxx = [list(bt.columns) for bt in bt_list]
#    allcols = np.unique([item for sublist in xxx for item in sublist])    
#    for i in range(len(bt_list)):
#        bt_list[i] = bt_list[i].reindex(columns=allcols)    
    bt_all = pd.concat(bt_list, ignore_index=True)

    # replace nan in boolean columns with False
    bt_bool = bt_all.select_dtypes('O').columns
    bt_all[bt_bool] = bt_all[bt_bool].fillna(False)

    def fillnap(bt_all, bt_bool):
        bt_all[bt_bool] = bt_all[bt_bool].fillna(False)
        return bt_all
    
    col = bt_all.columns
    n=int(bt_all.shape[0]/n_jobs)+1
    chunks = [bt_all[i:i+n].copy() for i in range(0,bt_all.shape[0],n)]
    bt_all = Parallel(n_jobs=n_jobs)(delayed(fillnap)(chunk, bt_bool) for chunk in chunks)
    bt_all = np.concatenate(bt_all)
    bt_all = pd.DataFrame(bt_all, columns = col)

    bt_all = bt_ckd(site, yearX, bt_all)
    bt_all = bt_postprocess(site, yearX, bt_all)
    bt_all.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3pos_'+site+'_'+stg+'_3000.pkl')
    

def combinebtpos(site, yearX, stg, threshold=0.01, n_jobs=15):

    onset = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_onset_'+site+'.pkl')
    years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
    bt_list = list()

    for year in years:
        try:
            data = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+str(year)+'.pkl')
            data = flag_convert(data, stg)
            # col = data.columns
            # n=int(data.shape[0]/n_jobs)+1
            # chunks = [data[i:i+n].copy() for i in range(0,data.shape[0],n)]            
            # bt_list = bt_list+chunks
            bt_list.append(data.copy())            
        except:
            print(str(year)+' not exists')

    bt_list = drop_too_much_nan_positive_parallel(site, yearX, bt_list, threshold, n_jobs=n_jobs)

    type_list = [dict(bt.dtypes) for bt in bt_list]
    type_ref = type_list[0].copy()
    for i in range(len(type_list)):
        type_ref.update(type_list[i])

    bool_feature = list()
    nonbool_feature = list()
    for k, v in type_ref.items():
        if v == bool:
            bool_feature.append(k)
        else:
            nonbool_feature.append(k)

    xxx = [list(bt.columns) for bt in bt_list]
    allcols = np.unique([item for sublist in xxx for item in sublist])    

    def df_add_column(bt, bool_feature, allcols):
        new_bool = [x for x in bool_feature if x not in bt.columns]
        new_bool_df = bt.reindex(columns=new_bool, fill_value=False)
        bt = pd.concat([bt.T,new_bool_df.T]).T
        bt = bt.reindex(columns=allcols)
        return bt
        
    bt_list = Parallel(n_jobs=n_jobs)(delayed(df_add_column)(bt, bool_feature, allcols) for bt in bt_list)      
    bt_all = pd.concat(bt_list, ignore_index=True)
    
    bt_all = bt_ckd(site, yearX, bt_all)
    bt_all = bt_postprocess(site, yearX, bt_all)
    bt_all = correct_dtypes(bt_all, site, stg)

    bt_all.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3pos_'+site+'_'+stg+'_3000.pkl')
    
if __name__ == "__main__":
    combinebtpos('KUMC', 3000, 'stg01', threshold=0.01, n_jobs=20)
    combinebtpos('KUMC', 3000, 'stg23', threshold=0.01, n_jobs=20)
    combinebtpos('KUMC', 3000, 'stg123', threshold=0.01, n_jobs=20)    