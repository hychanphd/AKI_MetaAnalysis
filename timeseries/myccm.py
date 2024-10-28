import pandas as pd
import numpy as np
import itertools
from scipy.stats import fisher_exact
import shelve
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
import csv
from dfply import *
import itertools
import os
import logging
from sys import getsizeof
import sklearn
import time
from sklearn.metrics import roc_auc_score
from catboost import Pool, cv
import xgboost
import catboost
import scipy.stats as st

import importlib
from joblib import Parallel, delayed
from joblib import parallel_backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
import scipy
import datetime
import pyEDM
from scipy import interpolate
from scipy.stats import pearsonr
import pyunicorn.timeseries
from scipy.linalg import lstsq
from joblib import Parallel, delayed
from scipy import optimize

from IPython.display import HTML
from itertools import combinations, product

import pandas as pd
import numpy as np
import itertools
from scipy.stats import fisher_exact
import shelve
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
#import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
import csv
from dfply import *
import itertools
import os
import logging
from sys import getsizeof
import sklearn
import time
from sklearn.metrics import roc_auc_score
from catboost import Pool, cv
import xgboost
import catboost
import scipy.stats as st

import importlib
from joblib import Parallel, delayed
from joblib import parallel_backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
import scipy
import datetime

from joblib import Parallel, delayed

import os
import statsmodels



class myccm:
    
    def __init__(self):
        self.loinc1 = None
        self.loinc2 = None    
        self.site = None
        self.labxx = None
        self.data_count = None
        self.raws = None
        self.rawr = None
        self.lab = None
        self.onset = None
        self.data_count = None
        self.raws = None
        self.rawr = None
        self.tentmap_df = None
        self.E1 = None
        self.E2 = None
        self.tau1 = None
        self.tau2 = None
        self.r1p = None
        self.r2p = None
        self.rphodf1 = None
        self.rphodf2 = None 
        self.data_count_filtered = None
        self.r1s = None
        self.r2s = None
        self.r1sp = None
        self.r2sp = None 
        self.y1f = None
        self.y2f = None
        self.scale = None
        self.diff_num1 = 0
        self.diff_num2 = 0
        self.PATID = None
        self.ENCOUNTERID = None
        self.datafolder = '/home/hoyinchan/blue/Data/data2022/'
        
    def copy(self, extobj):
        self.loinc1 = extobj.loinc1
        self.loinc2 = extobj.loinc2
        self.site = extobj.site
        self.labX = extobj.labX
        self.labxx = extobj.labxx
        self.data_count = extobj.data_count
        self.raws = extobj.raws
        self.rawr = extobj.rawr
        self.lab = extobj.lab
        self.onset = extobj.onset
        self.data_count = extobj.data_count
        self.data_count_filtered = extobj.data_count_filtered
        self.raws = extobj.raws
        self.rawr = extobj.rawr
        self.tentmap_df = extobj.tentmap_df
        self.E1 = extobj.E1
        self.E2 = extobj.E2
        self.tau1 = extobj.tau1
        self.tau2 = extobj.tau2
        self.r1p = extobj.r1p
        self.r2p = extobj.r2p
        self.rphodf1 = extobj.rphodf1
        self.rphodf2 = extobj.rphodf2

        
    # p<0.05 = non-stationary
    def kpss_test(self, data_df=None):
        if data_df is None:
            data_df=self.tentmap_df.drop('Time',axis=1)
        
        test_stat, p_val = [], []
        cv_1pct, cv_2p5pct, cv_5pct, cv_10pct = [], [], [], []
        for c in data_df.columns: 
            kpss_res = kpss(data_df[c].dropna(), regression='ct')
            test_stat.append(kpss_res[0])
            p_val.append(kpss_res[1])
            cv_1pct.append(kpss_res[3]['1%'])
            cv_2p5pct.append(kpss_res[3]['2.5%'])
            cv_5pct.append(kpss_res[3]['5%'])
            cv_10pct.append(kpss_res[3]['10%'])
        kpss_res_df = pd.DataFrame({'Test statistic': test_stat, 
                                   'p-value': p_val, 
                                   'Critical value - 1%': cv_1pct,
                                   'Critical value - 2.5%': cv_2p5pct,
                                   'Critical value - 5%': cv_5pct,
                                   'Critical value - 10%': cv_10pct}, 
                                 index=data_df.columns).T
        kpss_res_df = kpss_res_df.round(4)       
        kpss_res_df.loc['stationary',:] = kpss_res_df.loc['p-value',:]>=0.05        
        return kpss_res_df

    # p>0.05 = non-stationary    
    def adf_test(self, data_df=None):
        if data_df is None:
            data_df=self.tentmap_df.drop('Time',axis=1)        
        test_stat, p_val = [], []
        cv_1pct, cv_5pct, cv_10pct = [], [], []
        for c in data_df.columns: 
            adf_res = adfuller(data_df[c].dropna())
            test_stat.append(adf_res[0])
            p_val.append(adf_res[1])
            cv_1pct.append(adf_res[4]['1%'])
            cv_5pct.append(adf_res[4]['5%'])
            cv_10pct.append(adf_res[4]['10%'])
        adf_res_df = pd.DataFrame({'Test statistic': test_stat, 
                                   'p-value': p_val, 
                                   'Critical value - 1%': cv_1pct,
                                   'Critical value - 5%': cv_5pct,
                                   'Critical value - 10%': cv_10pct}, 
                                 index=data_df.columns).T
        adf_res_df = adf_res_df.round(4)
        adf_res_df.loc['stationary',:] = adf_res_df.loc['p-value',:]<0.05
        return adf_res_df        

    def stationarize(self, mode='log', windows=2, df = None):      
        if df is None:
            df = self.tentmap_df
        if mode == 'diff':
#            tentmap_df_tmp = df.drop('Time',axis=1,errors='ignore').diff().dropna().reset_index(drop=True).reset_index().rename({'index':'Time'},axis=1)
            tentmap_df_tmp = df.drop('Time',axis=1,errors='ignore').diff()
        elif mode == 'sqrt':
            tentmap_df_tmp = np.sqrt(df.drop('Time',axis=1,errors='ignore')).dropna().reset_index(drop=True).reset_index().rename({'index':'Time'},axis=1)
        elif mode == 'cbrt':
            tentmap_df_tmp = np.cbrt(df.drop('Time',axis=1,errors='ignore')).dropna().reset_index(drop=True).reset_index().rename({'index':'Time'},axis=1)
        elif mode == 'log':
            tentmap_df_tmp = np.log(df.drop('Time',axis=1,errors='ignore')).dropna().reset_index(drop=True).reset_index().rename({'index':'Time'},axis=1)
        elif mode == 'rollmean':
            tentmap_df_tmp = (df-df.rolling(window = windows).mean())
        return tentmap_df_tmp

    def stationarize_indv(self):
        tentmap_df_tmp1 = self.tentmap_df[[self.loinc1]]
        self.diff_num1 = 0
        for i in range(1,20):
            #stationary
            adf = self.adf_test(data_df=tentmap_df_tmp1)
            kpss = self.kpss_test(data_df=tentmap_df_tmp1)
            # print(i)
            # print(adf.loc['p-value',:])
            # print(kpss.loc['p-value',:])
            # print(adf.loc['stationary',:])
            # print(kpss.loc['stationary',:])        
            if np.all([adf.loc['stationary',:], kpss.loc['stationary',:]]):
                break
            self.diff_num1 = i
            tentmap_df_tmp1 = self.stationarize(mode='rollmean', windows=i+1, df=self.tentmap_df[[self.loinc1]])

        tentmap_df_tmp2 = self.tentmap_df[[self.loinc2]]
        self.diff_num2 = 0        
        for i in range(1,20):
            #stationary
            adf = self.adf_test(data_df=tentmap_df_tmp2)
            kpss = self.kpss_test(data_df=tentmap_df_tmp2)
            # print(i)
            # print(adf.loc['p-value',:])
            # print(kpss.loc['p-value',:])
            # print(adf.loc['stationary',:])
            # print(kpss.loc['stationary',:])            
            if np.all([adf.loc['stationary',:], kpss.loc['stationary',:]]):
                break
            self.diff_num2 = i                
            tentmap_df_tmp2 = self.stationarize(mode='rollmean', windows=i+1, df=self.tentmap_df[[self.loinc2]])   

        return tentmap_df_tmp1.merge(tentmap_df_tmp2, left_index=True, right_index=True, how='inner').dropna().reset_index(drop=True).reset_index().rename({'index':'Time'},axis=1)
        
    def get_data_2021(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        self.lab = pd.read_csv(datafolder+site+'/raw/AKI_LAB.csv')
        self.lab['SPECIMEN_DATE_TIME'] = pd.to_datetime(self.lab['SPECIMEN_DATE_TIME'])

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET', 'AKI2_ONSET', 'AKI3_ONSET']]
        x = x.bfill(axis=1)
        x = x[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']]
        x['AKI1_ONSET'] = pd.to_datetime(x['AKI1_ONSET'])        

#        xxx = self.labxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')
        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.labX = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(1, "d")]
#        self.labxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(0, "d")]
        self.labX = self.labX[self.labX['SPECIMEN_DATE_TIME']>=pd.to_datetime(self.labX['ADMIT_DATE'])]

    def get_vitaldata_2021(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        self.lab = pd.read_csv(datafolder+site+'/raw/AKI_VITAL.csv')
        self.lab['MEASURE_DATE_TIME'] = pd.to_datetime(self.lab['MEASURE_DATE_TIME'])

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET', 'AKI2_ONSET', 'AKI3_ONSET']]
        x = x.bfill(axis=1)
        x = x[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']]
        x['AKI1_ONSET'] = pd.to_datetime(x['AKI1_ONSET'])        

#        xxx = self.labxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')
        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.labV = xxx[xxx['AKI1_ONSET']-xxx['MEASURE_DATE_TIME']>=pd.Timedelta(1, "d")]
#        self.labxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(0, "d")]
        self.labV = self.labV[self.labV['MEASURE_DATE_TIME']>=pd.to_datetime(self.labV['ADMIT_DATE'])]
    
    
    
    def get_onsetdata_2021(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        onset = pd.read_csv(datafolder+site+'/raw/AKI_ONSETS.csv') 
        self.onset = onset[onset['NONAKI_SINCE_ADMIT'].isnull()]

        
        
    def get_data(self, loincs_cofounder_potassium, loincs_potassium, loincs_dependents, site='UTHSCSA'):
        self.site = site
        
        self.lab = pd.read_parquet(self.datafolder+site+'/p0_lab_g_'+site+'.parquet', columns=['PATID', 'ENCOUNTERID', 'ONSETS_ENCOUNTERID',  'SPECIMEN_DATE', 'RESULT_NUM', 'LAB_LOINC', 'DAYS_SINCE_ADMIT']) 
        self.lab['PATID'] = self.lab['PATID'].astype(str)
        self.lab['ENCOUNTERID'] = self.lab['ENCOUNTERID'].astype(str)
        self.lab['ONSETS_ENCOUNTERID'] = self.lab['ONSETS_ENCOUNTERID'].astype(str)        
        filter_lab = loincs_cofounder_potassium+loincs_potassium+loincs_dependents
        self.lab = self.lab[self.lab['LAB_LOINC'].isin(filter_lab)]
        
        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'ONSET_DATE']]
        x.columns = ['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']

        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')

        self.labV = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE']>=pd.Timedelta(1, "d")]
        self.labV = self.labV[self.labV['SPECIMEN_DATE']>=pd.to_datetime(self.labV['ADMIT_DATE'])]
        

    def get_vitaldata(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2022/'
        self.lab = pd.read_parquet(datafolder+site+'/p0_vital_old_'+site+'.parquet', columns=['PATID', 'ENCOUNTERID', 'ONSETS_ENCOUNTERID',  'MEASURE_DATE', 'DIASTOLIC', 'SYSTOLIC', 'DAYS_SINCE_ADMIT'])
        self.lab = self.lab.dropna()
        self.lab['MEASURE_DATE'] = pd.to_datetime(self.lab['MEASURE_DATE'])
        self.lab['PATID'] = self.lab['PATID'].astype(str)
        self.lab['ENCOUNTERID'] = self.lab['ENCOUNTERID'].astype(str)   
        self.lab['ONSETS_ENCOUNTERID'] = self.lab['ONSETS_ENCOUNTERID'].astype(str)        
        

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'ONSET_DATE']]
        x.columns = ['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']  

#        xxx = self.labxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')
        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.labV = xxx[xxx['AKI1_ONSET']-xxx['MEASURE_DATE']>=pd.Timedelta(1, "d")]
#        self.labxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(0, "d")]
        self.labV = self.labV[self.labV['MEASURE_DATE']>=pd.to_datetime(self.labV['ADMIT_DATE'])]
    
    
    
    def get_onsetdata(self, site='UTHSCSA'):
        self.site = site
        
        self.onset = pd.read_parquet(self.datafolder+site+'/p0_onset_'+site+'.parquet') 
        self.onset = self.onset.drop_duplicates()
        self.covid = pd.read_parquet(self.datafolder+site+'/p0_covid_status_'+site+'.parquet')
        self.covid = self.covid.drop_duplicates()

        self.onset['PATID'] = self.onset['PATID'].astype(str)
        self.onset['ENCOUNTERID'] = self.onset['ENCOUNTERID'].astype(str)
        self.covid['PATID'] = self.covid['PATID'].astype(str)
        self.covid['ENCOUNTERID'] = self.covid['ENCOUNTERID'].astype(str)
        self.onset = self.covid[self.covid['BCCOVID']][['PATID', 'ENCOUNTERID']].merge(self.onset, on = ['PATID', 'ENCOUNTERID'], how='left')
        
        
        
        
        
    def set_loincs_pair(self, loinc1 = None, loinc2 = None):
        self.loinc1 = loinc1
        self.loinc2 = loinc2          
        
    def extract_time_series_pair(self):
        labx = self.labX[(self.labX['LAB_LOINC']==self.loinc1) | (self.labX['LAB_LOINC']==self.loinc2)]
        labx = labx[['PATID','ENCOUNTERID','SPECIMEN_DATE_TIME','LAB_LOINC','RESULT_NUM']]
        labx = labx.groupby(['PATID','ENCOUNTERID','SPECIMEN_DATE_TIME','LAB_LOINC']).mean().reset_index()
        self.labxx = labx.pivot(index=['PATID','ENCOUNTERID','SPECIMEN_DATE_TIME'], columns = 'LAB_LOINC', values='RESULT_NUM').reset_index()
        
        self.data_count = self.labxx.groupby(['PATID','ENCOUNTERID']).count().sort_values('SPECIMEN_DATE_TIME').reset_index()
        self.data_count_filtered = self.data_count[(self.data_count[self.loinc1] >=30) & (self.data_count[self.loinc2] >=30)].sort_values(self.loinc1)

    def select_paitient(self, pos=-1):
        self.pos = pos
        tmp = self.labxx[self.labxx['ENCOUNTERID']==self.data_count_filtered['ENCOUNTERID'].iloc[pos]]
        self.PATID = tmp.iloc[0,:]['PATID']
        self.ENCOUNTERID = str(tmp.iloc[0,:]['ENCOUNTERID'])
        tmp = tmp.sort_values('SPECIMEN_DATE_TIME')
        tmp = tmp.reset_index(drop=True)
        fvi = max(tmp[self.loinc1].first_valid_index(), tmp[self.loinc2].first_valid_index())
        lvi = min(tmp[self.loinc1].last_valid_index(), tmp[self.loinc2].last_valid_index())
        tmp = tmp[fvi:lvi+1]
        tmp.index=tmp['SPECIMEN_DATE_TIME']
        self.raws = tmp[self.loinc1]
        self.rawr = tmp[self.loinc2]           
    
    def regular_ts_interpolate(self, s, td):
        oidx = s.index
        nidx = pd.date_range(oidx.min(), oidx.max(), freq=td)
        res = s.reindex(oidx.union(nidx)).interpolate('index').reindex(nidx)
        return res    
    
    def regular_ts_bin(self, s, td):
        df_group = s.copy().groupby(pd.Grouper(level='SPECIMEN_DATE_TIME', freq=td)).agg('mean')   
#        print(df_group.isnull().sum()/df_group.shape[0])
        df_group = df_group.interpolate()
        return df_group.interpolate()

    def split_time_series(self, s, td):
        df_group = s.copy().groupby(pd.Grouper(level='SPECIMEN_DATE_TIME', freq=td)).agg('mean')
        xxx = df_group.reset_index().reset_index()
        yyy = df_group.reset_index().reset_index()
        xxx = xxx[xxx[xxx.columns[-1]].isnull()]
        xxx2 = xxx['index'].diff()
        xxx3 = xxx2.reset_index()
        
        splitpt = xxx3[xxx3['index']==1]['level_0']
        splitpt2 = [-1] + list(splitpt) + [df_group.shape[0]+2]
        dfsplit = [yyy[(splitpt2[i]+1):(splitpt2[i+1]-1)] for i in range(0, len(splitpt2)-1)]
        if len(dfsplit) > 1:
            dfsplit2 = np.array(dfsplit)[[len(x)>=5 for x in dfsplit]]
        else:
            dfsplit2 = dfsplit
        dfsplit3 = [x.interpolate() for x in dfsplit2] 
        return dfsplit3, [x.iloc[:,-1].shape[0] for x in dfsplit2], [x.iloc[:,-1].isnull().sum()/x.iloc[:,-1].shape[0] for x in dfsplit2]
    
    def split_time_series_2d(self, s, r, td):
        def before_split(s, td):
            df_group = s.copy().groupby(pd.Grouper(level='SPECIMEN_DATE_TIME', freq=td)).agg('mean')
            xxx = df_group.reset_index().reset_index()
            yyy = df_group.reset_index().reset_index()
            xxx = xxx[xxx[xxx.columns[-1]].isnull()]
            xxx2 = xxx['index'].diff()
            xxx3 = xxx2.reset_index()
            splitpt = list(xxx3[xxx3['index']==1]['level_0'])
            return df_group, yyy, splitpt

        df_groups, yyys, splitpts = before_split(s, td)
        df_groupr, yyyr, splitptr = before_split(r, td)
        splitpt = np.unique(np.sort(splitpts+splitptr))
        
        def after_split(df_group, yyy, splitpt):
            splitpt2 = [-1] + list(splitpt) + [df_group.shape[0]+2]
            dfsplit = [yyy[(splitpt2[i]+1):(splitpt2[i+1]-1)] for i in range(0, len(splitpt2)-1)]
            if len(dfsplit) > 1:
                dfsplit2 = np.array(dfsplit)[[len(x)>=5 for x in dfsplit]]                
#                dfsplit2 = np.array(dfsplit)[[len(x)>=1 for x in dfsplit]]
            else:
                dfsplit2 = dfsplit
            dfsplit3 = [x.interpolate() for x in dfsplit2] 
            return dfsplit3, [x.iloc[:,-1].shape[0] for x in dfsplit2], [x.iloc[:,-1].isnull().sum()/x.iloc[:,-1].shape[0] for x in dfsplit2]
        
        dfs, lens, nanratios = after_split(df_groups, yyys, splitpt)
        dfr, lenr, nanratior = after_split(df_groupr, yyyr, splitpt)
        return dfs, lens, nanratios, dfr, lenr, nanratior
    
    def interpolate_time_series(self, scale=1, mode='split', pos=None, plotts=False):
        tds = np.diff(np.sort(self.raws.dropna().index)).mean().astype('timedelta64[h]')
        tdr = np.diff(np.sort(self.rawr.dropna().index)).mean().astype('timedelta64[h]')
        tdmax = max(tds, tdr)

        td = (tdmax*scale).astype(int).astype(str) + ' H'

        if mode=='bin':
            s = self.regular_ts_bin(self.raws, td)
            r = self.regular_ts_bin(self.rawr, td)        
        elif mode == 'interpolate':
            s = self.regular_ts_interpolate(self.raws, td)
            r = self.regular_ts_interpolate(self.rawr, td)
        elif mode == 'split1d':
            s, lens, nanratios = self.split_time_series(self.raws, td)
            if pos is None:
                # print(lens)
                # print(nanratios)
                indsmax = np.array([len(x) for x in s]).argmax()
                self.data_ratio= len(s[indsmax])/np.sum([len(x) for x in s])   
                s = s[indsmax]
                self.nanratios = nanratios[indsmax]
            else:
                s = s[pos]
            s.index = s['SPECIMEN_DATE_TIME']  
            s = s.iloc[:,-1]
            
            r, lenr, nanratior = self.split_time_series(self.rawr, td)
            if pos is None:
                indrmax = np.array([len(x) for x in r]).argmax()
                r = r[indsmax]
                self.nanratior = nanratior[indrmax]
            else:
                r = r[pos]            
            r.index = r['SPECIMEN_DATE_TIME']                        
            r = r.iloc[:,-1]
        elif mode == 'split':
            s, lens, nanratios, r, lenr, nanratior = self.split_time_series_2d(self.raws, self.rawr, td)
            if pos is None:
                indmax = np.array([len(x) for x in r]).argmax()
                self.data_ratio= len(s[indmax])/np.sum([len(x) for x in s])                                
                s = s[indmax]
                r = r[indmax]
                self.nanratios = nanratios[indmax]                
                self.nanratior = nanratior[indmax]
                s.index = s['SPECIMEN_DATE_TIME']  
                s = s.iloc[:,-1]                
                r.index = r['SPECIMEN_DATE_TIME']  
                r = r.iloc[:,-1]
            else:
                r = r[pos]   
                s = s[pos]
        else:
            s = self.raws
            r = self.rawr

        if plotts:
            r.plot(style='.-')
            s.plot(style='.-')        
            self.raws.plot(style='o')             
            self.rawr.plot(style='o')   
        
        self.tentmap_df = pd.DataFrame([s, r]).T
        self.tentmap_df = self.tentmap_df.reset_index(drop=True).reset_index()
        self.tentmap_df.columns = ['Time',self.loinc1, self.loinc2]
        
    def get_optimal_embedding(self, Tps=[1]):
        optimal_emdeddings = list()
        lx = self.tentmap_df.shape[0]
        lib = "1 "+str(np.floor(lx/5))
        pred = str(np.floor(lx/5)) + " " + str(lx)
        for Tp in Tps:
            for tau in range(-20,0):
                for E in range(1,20):
                    try:
                        x1 = pyEDM.Simplex(dataFrame=self.tentmap_df, lib=lib , pred=pred, columns=self.loinc1, target=self.loinc1, Tp=Tp, tau=tau, E=E)
                        optimal_emdedding = pd.DataFrame([E, pyEDM.ComputeError(x1['Observations'], x1['Predictions'])['rho']]).T
                        optimal_emdedding.columns = ['E', 'rho']
            #            optimal_emdedding = pyEDM.EmbedDimension(dataFrame=self.tentmap_df, lib=lib , pred=pred, columns='calcium', target='calcium', Tp=1, tau=tau, maxE=7)
                        optimal_emdedding['tau'] = tau
                        optimal_emdedding['Tp'] = Tp            
                        optimal_emdedding['target'] = self.loinc1
                        optimal_emdeddings.append(optimal_emdedding)        

                        x1 = pyEDM.Simplex(dataFrame=self.tentmap_df, lib=lib , pred=pred, columns=self.loinc2, target=self.loinc2, Tp=Tp, tau=tau, E=E)
                        optimal_emdedding = pd.DataFrame([E, pyEDM.ComputeError(x1['Observations'], x1['Predictions'])['rho']]).T
                        optimal_emdedding.columns = ['E', 'rho']            
            #            optimal_emdedding = pyEDM.EmbedDimension(dataFrame=self.tentmap_df, lib=lib , pred=pred, columns='scr', target='scr', Tp=1, tau=tau, maxE=7)
                        optimal_emdedding['tau'] = tau     
                        optimal_emdedding['target'] = self.loinc2        
                        optimal_emdedding['Tp'] = Tp                            
                        optimal_emdeddings.append(optimal_emdedding)
                    except:
                        pass
        self.optimal_emdeddings = pd.concat(optimal_emdeddings)
        optimal_values = self.optimal_emdeddings.sort_values(['rho','tau'],ascending=False).groupby('target').head(1)        
#        print(optimal_values)
        self.E1 = int(optimal_values[optimal_values['target']==self.loinc1]['E'].iloc[0])
        self.E2 = int(optimal_values[optimal_values['target']==self.loinc2]['E'].iloc[0])
        self.tau1 = int(optimal_values[optimal_values['target']==self.loinc1]['tau'].iloc[0])
        self.tau2 = int(optimal_values[optimal_values['target']==self.loinc2]['tau'].iloc[0])
#        print([self.E1, self.E2, self.tau1, self.tau2])

    def gen_surr_series(self):
        surr_df = self.tentmap_df.iloc[permut,:]
        surr_df['Time'] = list(range(self.tentmap_df.shape[0]))
        return surr_df

    def gen_ccm_stat(self, r):
        def ttest(a):
            return scipy.stats.ttest_1samp(a, 0).pvalue
        def q05(x):
            return x.quantile(0.05)
        def q95(x):
            return x.quantile(0.95)  
        return r[['E','nn','tau','LibSize','rho']].groupby(['E','nn','tau','LibSize']).agg([np.mean, np.std, ttest, q05, q95]).reset_index()
    
    def run_ccm(self ,df=None,sample=10, step=5, Tp=0, maxlibSize=None):        
        dfnonef = False
        if df is None:
            dfnonef = True
            df = self.tentmap_df

        if maxlibSize is None:
            maxlibSize1 = self.maxallowedlibSize(self.E1, self.tau1, Tp)
            maxlibSize2 = self.maxallowedlibSize(self.E2, self.tau2, Tp)
        else:
            maxlibSize1 = maxlibSize
            maxlibSize2 = maxlibSize            
            
        libSizes1 = ' '.join([str(x) for x in [max(10, 2*self.E1), maxlibSize1, step]])
        libSizes2 = ' '.join([str(x) for x in [max(10, 2*self.E2), maxlibSize2, step]])
        
        r1 = pyEDM.CCM(dataFrame=df, columns=self.loinc1, target=self.loinc2, E=self.E1, Tp=Tp, tau=self.tau1, sample = sample, libSizes=libSizes1, replacement=True, includeData=True)
        r2 = pyEDM.CCM(dataFrame=df, columns=self.loinc2, target=self.loinc1, E=self.E2, Tp=Tp, tau=self.tau2, sample = sample, libSizes=libSizes2, replacement=True, includeData=True)
      
        if dfnonef:
            r1p = self.gen_ccm_stat(r1['PredictStats1'])
            r2p = self.gen_ccm_stat(r2['PredictStats1'])            
            [self.r1, self.r2, self.r1p, self.r2p] = [r1, r2, r1p, r2p]
        else:
            return r1, r2
        
    def plot_convergence(self, r1p=None, r2p=None, figsize=(16, 12)):
        plt.figure(figsize=figsize)
        r1pnone = False
        if r1p is None:
            r1pnone = True
            r1p=self.r1p
            r2p=self.r2p
        
        self.calculate_correlation()
        label = self.loinc1+' xmap '+self.loinc2
        plt.plot(r1p['LibSize'], r1p[('rho','mean')], 'o', label = label, color='orange')
#        plt.fill_between(r1p['LibSize'], r1p[('rho','mean')]-r1p[('rho','std')], r1p[('rho','mean')]+r1p[('rho','std')], alpha=0.1)
        
        label = self.loinc2+' xmap '+self.loinc1
        plt.plot(r2p['LibSize'], r2p[('rho','mean')], 'o', label = label, color='blue')
#        plt.fill_between(r2p['LibSize'], r2p[('rho','mean')]-r2p[('rho','std')], r2p[('rho','mean')]+r2p[('rho','std')], alpha=0.1)

        plt.axhline(y=0, color='g')
        plt.axhline(y=np.abs(self.corr), linestyle='--', label = 'correlation', color='black')
        if self.y1fg is not None and r1pnone:
            plt.plot(r1p['LibSize'], r1p['rhof'], label=self.loinc1+':rho_\inf='+str(np.round(self.y1fg[0],2))+':exp='+str(np.round(self.y1fg[2],2))+':r^2='+str(np.round(self.y1fg[-1],2)), color='orange')
        if self.y2f is not None and r1pnone:
            plt.plot(r2p['LibSize'], r2p['rhof'], label=self.loinc2+':rho_\inf='+str(np.round(self.y2fg[0],2))+':exp='+str(np.round(self.y2fg[2],2))+':r^2='+str(np.round(self.y2fg[-1],2)), color='blue')
        if self.r1s is not None and r1pnone:
            m1, m2 = self.max_surrogate_q95()
            plt.axhline(y=m1, linestyle='--', label = self.loinc1+'_q95', color='orange')
            plt.axhline(y=m2, linestyle='--', label = self.loinc2+'_q95', color='blue')
            plt.fill_between(self.r1sp['LibSize'], self.r1sp[('rho','q05')], self.r1sp[('rho','q95')], alpha=0.1, color='orange')
            plt.fill_between(self.r2sp['LibSize'], self.r2sp[('rho','q05')], self.r2sp[('rho','q95')], alpha=0.1, color='blue')            
        else:
            plt.fill_between(r1p['LibSize'], r1p[('rho','q05')], r1p[('rho','q95')], alpha=0.1, color='orange')
            plt.fill_between(r2p['LibSize'], r2p[('rho','q05')], r2p[('rho','q95')], alpha=0.1, color='blue')

        plt.legend()
        plt.show()
    
    def gen_delay(self, sample=10, range_lim=10, libSizes=None, E1=None, E2=None, tau1=None, tau2=None):
        Tps = list(range(range_lim*-1,range_lim))
        rphos = list()
#        libSizes = "5"      
        if E1 == None:
            E1 = self.E1
        if E2 == None:
            E2 = self.E2
        if tau1 == None:
            tau1 = self.tau1
        if tau2 == None:
            tau2 = self.tau2
            
        for Tp in Tps: 
            if libSizes == None:
    #            libSizes = str(self.tentmap_df.shape[0])   
                 libSizes1 = str(self.maxallowedlibSize(E1, tau1, Tp))
                 libSizes2 = str(self.maxallowedlibSize(E2, tau2, Tp))
            else:
                 libSizes1 = libSizes
                 libSizes2 = libSizes
            rpho = pyEDM.CCM(dataFrame=self.tentmap_df, columns=self.loinc1, target=self.loinc2, E=E1, Tp=Tp, tau=tau1, sample = sample, libSizes=libSizes1, replacement=True, includeData=True)
            rpho['Tp'] = Tp
            rpho['source'] = self.loinc1
            rpho['target'] = self.loinc2            
            rphos.append(rpho)
            
            rpho = pyEDM.CCM(dataFrame=self.tentmap_df, columns=self.loinc2, target=self.loinc1, E=E2, Tp=Tp, tau=tau2, sample = sample, libSizes=libSizes2, replacement=True, includeData=True)
            rpho['Tp'] = Tp
            rpho['source'] = self.loinc2
            rpho['target'] = self.loinc1                        
            rphos.append(rpho)            
        
        self.rphos = rphos
        rphos1 = [x for x in rphos if x['source']==self.loinc1]
        rphodf1 = pd.concat([x['LibMeans'] for x in rphos1])
        rphodf1['std'] = [x['PredictStats1']['rho'].std() for x in rphos1]
        rphodf1['Tp'] = [x['Tp'] for x in rphos1]
        
        rphos2 = [x for x in rphos if x['source']==self.loinc2]
        rphodf2 = pd.concat([x['LibMeans'] for x in rphos2])
        rphodf2['std'] = [x['PredictStats1']['rho'].std() for x in rphos2]
        rphodf2['Tp'] = [x['Tp'] for x in rphos2]
        
        self.rphodf1 = rphodf1
        self.rphodf2 = rphodf2
        
    def plot_delat(self, figsize=(16, 12)):
        plt.figure(figsize=figsize)
        self.calculate_correlation()
        col = self.loinc1+':'+self.loinc2
        label = self.loinc1+' xmap '+self.loinc2
        plt.plot(self.rphodf1['Tp'], self.rphodf1[col], label=label, color='orange')
        plt.fill_between(self.rphodf1['Tp'], self.rphodf1[col]-self.rphodf1['std'], self.rphodf1[col]+self.rphodf1['std'],alpha=0.1, color='orange')
        max1 = self.rphodf1.sort_values(col,ascending=False).head(1)
        plt.plot(max1['Tp'], max1[col], 'rx', markersize=20)
        if self.r1s is not None:
            m1, m2 = self.max_surrogate_q95()
            plt.axhline(y=m1, linestyle='--', label = self.loinc1+'_q95', color='orange')
        col = self.loinc2+':'+self.loinc1        
        label = self.loinc2+' xmap '+self.loinc1
        plt.plot(self.rphodf2['Tp'], self.rphodf2[col], label=label, color='blue')
        plt.fill_between(self.rphodf2['Tp'], self.rphodf2[col]-self.rphodf2['std'], self.rphodf2[col]+self.rphodf2['std'],alpha=0.1, color='blue')
        max1 = self.rphodf2.sort_values(col,ascending=False).head(1)
        plt.plot(max1['Tp'], max1[col], 'rx', markersize=20)        
        if self.r1s is not None:
            m1, m2 = self.max_surrogate_q95()
            plt.axhline(y=m2, linestyle='--', label = self.loinc1+'_q95', color='blue')        
        plt.axvline(x=0, color='r')
        plt.axhline(y=0, color='g')
        plt.axhline(y=np.abs(self.corr), linestyle='--', color='black')
        plt.legend()
    
    def plot_delat_smooth(self, which=1):
        if which == 1:
            loinc1 = self.loinc1
            loinc2 = self.loinc2
            col = loinc1+':'+loinc2   
            label = loinc1+' xmap '+loinc2            
            plt.plot(self.rphodf1['Tp'], self.rphodf1[col], color='orange')
            plt.errorbar(self.rphodf1['Tp'], self.rphodf1[col], yerr=self.rphodf1['std'], solid_capstyle='projecting', capsize=5, color='orange')
        if self.r1s is not None and r1pnone:
            m1, m2 = self.max_surrogate_q95()
            plt.axhline(y=m1, linestyle='--', label = self.loinc1+'_q95', color='orange')
        else:
            loinc2 = self.loinc1
            loinc1 = self.loinc2    
            col = loinc1+':'+loinc2   
            label = loinc1+' xmap '+loinc2            
            plt.plot(self.rphodf2['Tp'], self.rphodf2[col], color='blue')
            plt.errorbar(self.rphodf2['Tp'], self.rphodf2[col], yerr=self.rphodf2['std'], solid_capstyle='projecting', capsize=5, color='blue')
        if self.r1s is not None and r1pnone:
            m1, m2 = self.max_surrogate_q95()
            plt.axhline(y=m2, linestyle='--', label = self.loinc1+'_q95', color='blue')
        def processpre(df, Tp):
            df['Tp'] = Tp
            return df

        rphos1 = [x for x in self.rphos if x['source']==loinc1]
        rphodf1 = pd.concat([processpre(x['PredictStats1'], x['Tp']) for x in rphos1])

        col = loinc1+':'+loinc2   
        label = loinc1+' xmap '+loinc2
        tck,u = interpolate.splprep([rphodf1['Tp'], rphodf1['rho']],k=5,s=20)
        out = interpolate.splev(u,tck)

        plt.plot(out[0], out[1], 'b', label=label)
        

        #cs = CubicSpline(rphodf1['Tp'], rphodf1['rho'])
        plt.axvline(x=0, color='r')
        plt.axhline(y=0, color='g')
        plt.legend()
        
    def calculate_correlation(self):
        self.corr, _ = pearsonr(self.tentmap_df.iloc[:,1], self.tentmap_df.iloc[:,2])
        
    def mysurrogate(self):
        inputdf = self.tentmap_df
        df = inputdf.iloc[:,1:]
        surrogare_ts = pyunicorn.timeseries.Surrogates(self.tentmap_df.iloc[:,1:].T.to_numpy(), silence_level=2)        
        df_surr = pd.DataFrame(surrogare_ts.AAFT_surrogates(df.T.to_numpy())).T
        df_surr.columns = inputdf.columns[1:]
        df_surr.insert(0,'Time',inputdf['Time'])
        return df_surr
    
    def run_surrogate(self, sample=100, Tp=0):
        r1d = list()
        r2d = list()
        for i in range(100):
            surr_df = self.mysurrogate()
            r1, r2 = self.run_ccm(df=surr_df,sample=1,step=1, Tp=Tp)
            r1d.append(r1.copy())
            r2d.append(r2.copy())
        r1 = pd.concat([x['PredictStats1'] for x in r1d])
        r2 = pd.concat([x['PredictStats1'] for x in r2d])        
        r1p = self.gen_ccm_stat(r1)
        r2p = self.gen_ccm_stat(r2)
        self.r1s = r1
        self.r2s = r2
        self.r1sp = r1p
        self.r2sp = r2p        
#        return r1, r2, r1p, r2p

    def max_surrogate_q95(self):
        return [max(self.r1sp[('rho','q95')]), max(self.r2sp[('rho','q95')])]
    
    def run_complete_per_paitient(self, pos=-6):
        self.select_paitient(pos=pos)
        self.interpolate_time_series(scale=1, mode='split', pos=1)
        self.get_optimal_embedding()
        self.run_ccm(sample=100, step=1, Tp=0)
        self.gen_delay(range_lim=10, sample=100)
        
    def fit_exp(self, x, y):
        M = np.empty(y.shape + (2,), dtype=y.dtype)
        np.subtract(x, x[0], out=M[:, 0])
        M[0, 1] = 0
        np.cumsum(0.5 * np.diff(x) * (y[1:] + y[:-1]), out=M[1:, 1])
        Y = y - y[0]
        (A, B), *_ = lstsq(M, Y, overwrite_a=True, overwrite_b=True)
        a, c = -A / B, B
        M[:, 0].fill(1.0)
        np.exp(c * x, out=M[:, 1])
        (a, b), *_ = lstsq(M, y, overwrite_a=True, overwrite_b=False)
        out = np.array([a, b, c])
        yf = self.plot_fit_exp(out, x)
        rsq = sklearn.metrics.r2_score(y, yf)
        out = np.append(out, rsq)
        return out
    
    def get_fit(self):
        def func(x, a, b, c):
            return a + b * np.exp(c * x)
        self.y1fg = self.fit_exp(np.array(self.r1p['LibSize']), np.array(self.r1p[('rho', 'mean')]))        
        try:
            popt, pcov = optimize.curve_fit(func, np.array(self.r1p['LibSize']), np.array(self.r1p[('rho', 'mean')]), self.y1fg[0:3])
            self.y1f = popt
            self.y1fe = np.sqrt(np.diag(pcov))
        except:
            self.y1f = np.array([None, None, None])
            self.y1fe = np.array([None, None, None])
        
        self.y2fg = self.fit_exp(np.array(self.r2p['LibSize']), np.array(self.r2p[('rho', 'mean')]))  
        try:
            popt, pcov = optimize.curve_fit(func, np.array(self.r2p['LibSize']), np.array(self.r2p[('rho', 'mean')]), self.y2fg[0:3])
            self.y2f = popt
            self.y2fe = np.sqrt(np.diag(pcov))
        except:
            self.y2f = np.array([None, None, None])
            self.y2fe = np.array([None, None, None])
            
        self.r1p['rhof'] = self.plot_fit_exp(self.y1fg, self.r1p['LibSize'])
        self.r2p['rhof'] = self.plot_fit_exp(self.y2fg, self.r2p['LibSize'])
    
    def plot_fit_exp(self, out, x):
        return out[0] + out[1] * np.exp(out[2] * x)
    
#    self.plot_convergence(r1p=self.r1sp, r2p=self.r2sp)

    def one_process(self, pos, rescale=False, stationarize=False, Tp=0, maxlibSize=None):
        self.df_row = pd.DataFrame([self.loinc1, self.loinc2, pos, None, None, None, None, None,None,None,None,None,None,None,None,None,None,None,None,None]).T
        self.df_row.columns = ['feature1', 'feature2', 'PATID', 'ENCOUNTERID', 'binning_scale', 'NaN_ratio', 'Data_ratio', 'Num_Diff', 'rho1_inf', 'rho1_rexponent', 'rho1_r^2', 'rho2_inf', 'rho2_rexponent', 'rho2_r^2', 'rho1_q95', 'rho2_q95', 'rho1_last', 'rho2_last', '1xmap2', '2xmap1']
        
#        try:
        self.select_paitient(pos)
        if rescale:            
            for i in range(20):
                self.scale = 1+i/10
                self.interpolate_time_series(scale=self.scale, mode='split', plotts=False)
                print([self.scale, self.nanratios, self.data_ratio])
                if self.nanratios <= 0.1:
                    break
        else:
            self.scale = 1
            self.interpolate_time_series(scale=self.scale, mode='split', plotts=False)
            print([self.scale, self.nanratios, self.data_ratio])

#        print('finish rescale')
        if stationarize:            
            self.stationarize_indv()
        else:
            i = 0

#        print('finish stationarize')
#        print('NumberofDiff='+str(i))
        self.get_optimal_embedding()
#        print('finish get_optimal_embedding')        
        self.run_ccm(sample=100, step=1, Tp=Tp, maxlibSize=maxlibSize)
#        print('finish run_ccm')                
        self.run_surrogate(Tp=Tp)
#        print('finish run_surrogate')                        
        self.get_fit()
        self.collect_data()
        return [self.df_row]
#        except:
#            return [self.df_row]
    
    def surr_stat_test(self):
        x = self.r1['PredictStats1']
        N = sorted(x['LibSize'].drop_duplicates(), reverse=True)[4]
        sample_real = x[x['LibSize']>=N]['rho']                
        sample_surr = self.r1s[self.r1s['LibSize']>=N]['rho']
#        stdummy, pvalue1 = scipy.stats.ttest_ind(sample_real, sample_surr)
        q95 = np.quantile(sample_surr, 0.95)
#        stdummy, pvalue1 = scipy.stats.ttest_1samp(sample_real, q95, alternative='greater')
        stdummy, pvalue1 = scipy.stats.ttest_1samp(sample_real, q95)
        mean1 = sample_real.mean() > q95
        
        x = self.r2['PredictStats1']
        N = sorted(x['LibSize'].drop_duplicates(), reverse=True)[4]
        sample_real = x[x['LibSize']>=N]['rho']                
        sample_surr = self.r2s[self.r2s['LibSize']>=N]['rho']
#        stdummy, pvalue2 = scipy.stats.ttest_ind(sample_real, sample_surr)
        q95 = np.quantile(sample_surr, 0.95)
#        stdummy, pvalue2 = scipy.stats.ttest_1samp(sample_real, q95, alternative='greater')
        stdummy, pvalue2 = scipy.stats.ttest_1samp(sample_real, q95)
        mean2 = sample_real.mean() > q95

        if mean1 == False:
            pvalue1 = 1
        if mean2 == False:
            pvalue2 = 1
            
        return pvalue1, pvalue2
    
    def collect_data(self):
        m1, m2 = self.max_surrogate_q95()
        lastrho1 = self.r1p[('rho','mean')].iloc[-1]
        lastrho2 = self.r2p[('rho','mean')].iloc[-1]
        lastread1 = self.tentmap_df[self.loinc1].iloc[-1]
        lastread2 = self.tentmap_df[self.loinc2].iloc[-1]        
        #r^2>0.5, exponent<0 (decay), rhoinf and rholast>q95
        pvalue1, pvalue2 = self.surr_stat_test()
        causal1 = self.y1fg[3]>0.5 and self.y1fg[2]<0 and self.y1fg[0]>m1 and pvalue1<0.05
        causal2 = self.y2fg[3]>0.5 and self.y2fg[2]<0 and self.y2fg[0]>m2 and pvalue2<0.05
        self.df_row = pd.DataFrame([self.site, self.loinc1, self.loinc2, self.PATID, self.ENCOUNTERID, self.scale, 
                                    self.nanratios, self.data_ratio, self.diff_num1, self.diff_num2, 
                                    self.E1, self.E2, self.tau1, self.tau2,                                     
                                    self.y1fg[0], self.y1fg[2], self.y1fg[3], self.y2fg[0], self.y2fg[2], self.y2fg[3], m1, m2, lastrho1, lastrho2, causal1, causal2, pvalue1, pvalue2, lastread1, lastread2]).T
        self.df_row.columns = ['site', 'feature1', 'feature2', 'PATID', 'ENCOUNTERID', 'binning_scale', 'NaN_ratio', 'Data_ratio', 'Num_Diff1', 'Num_Diff2', 
                               'E1', 'E2', 'tau1', 'tau2',
                               'rho1_inf', 'rho1_rexponent', 'rho1_r^2', 'rho2_inf', 'rho2_rexponent', 'rho2_r^2', 'rho1_q95', 'rho2_q95', 'rho1_last', 'rho2_last', '1xmap2', '2xmap1', 'pvalue1', 'pvalue2','last_loinc1_reading', 'last_loinc2_reading']
    
    def print_dfrow(self):
        return HTML(self.df_row.to_html())
    
    def maxallowedlibSize(self, E, tau, Tp):
        return self.tentmap_df.shape[0] - abs(tau) * (E-1) + (Tp+1)

    def one_process_parallel(self, pos):
        try:
#            print(pos)
            self.one_process(pos, rescale=True, stationarize=True)
            return [self.df_row]
        except:
            self.df_row = pd.DataFrame([None, None, None, pos, None, None, 
                                        None,None,None,None, 
                                        None,None,None,None,                                    
                                        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]).T
            self.df_row.columns = ['site', 'feature1', 'feature2', 'PATID', 'ENCOUNTERID', 'binning_scale', 'NaN_ratio', 'Data_ratio', 'Num_Diff1', 'Num_Diff2', 
                                   'E1', 'E2', 'tau1', 'tau2',
                                   'rho1_inf', 'rho1_rexponent', 'rho1_r^2', 'rho2_inf', 'rho2_rexponent', 'rho2_r^2', 'rho1_q95', 'rho2_q95', 'rho1_last', 'rho2_last', '1xmap2', '2xmap1', 'pvalue1', 'pvalue2','last_loinc1_reading', 'last_loinc2_reading']
            return [self.df_row]

    def combined_analysis(self):
        resultsfiles = [x for x in os.listdir('.') if 'result_' in x ]

        csvs = list()
        for file in resultsfiles:
            csvtemp = pd.read_csv(file)
            csvs.append(csvtemp)
        csvt = pd.concat(csvs)

        csvt = csvt.drop('Unnamed: 0',axis=1, errors='ignore').dropna()
        csvt = csvt[(csvt['Num_Diff1']!=19) & (csvt['Num_Diff2']!=19)]

        pvalue1_adj = statsmodels.stats.multitest.multipletests(csvt['pvalue1'])
        pvalue2_adj = statsmodels.stats.multitest.multipletests(csvt['pvalue2'])

        csvt['pvalue1_adj'] = pvalue1_adj[1]
        csvt['pvalue2_adj'] = pvalue2_adj[1]

        csvt['1xmap2_adj'] = csvt['1xmap2'] & (csvt['pvalue1_adj']<0.05)
        csvt['2xmap1_adj'] = csvt['2xmap1'] & (csvt['pvalue2_adj']<0.05)        
        return csvt
    
    def one_process_up(self, redo=False):
        loincs_targets = ['17861-6', #Calcium
                          '2160-0',  #sCr
                          '2823-3',  #Potassium
                          '2951-2']  #Sodium
        
        loincs_cofounder_Calcium = []
        loincs_cofounder_sodium = []
        loincs_cofounder_potassium = ['2157-6', #creatine kinase
                                      '1920-8', #AST      
                                      '2532-0', #LDH
                                      '4542-7', #Haptoglobin 
                                      '3084-1'] #uric acid
        loincs_potassium = ['2823-3', '2160-0']
        loincs_all_potassium = list(product(loincs_cofounder_potassium, loincs_potassium)) + list(combinations(loincs_potassium, 2))
        loincs_pairs = loincs_all_potassium

        for loincp in loincs_pairs:
            loinc1 = loincp[0]
            loinc2 = loincp[1]
            savepath = '/home/hoyinchan/code/AKI_CDM_PY/timeseries/result_'+self.site+'_'+loinc1.replace('-','_')+'_'+loinc2.replace('-','_')+'.csv'
            if not redo and not os.path.exists(savepath):
#                try:
                self.set_loincs_pair(loinc1=loinc1, loinc2=loinc2)
                self.extract_time_series_pair()
#                df_row = Parallel(n_jobs=8)(delayed(self.one_process_parallel)(pos) for pos in range(8))
                df_row = Parallel(n_jobs=63)(delayed(self.one_process_parallel)(pos) for pos in range(len(self.data_count_filtered['ENCOUNTERID'])))        
                pd.concat([x[0] for x in df_row]).to_csv(savepath)
#                except:
#                    pass    
    
########################################################################################################################################################
# pco


    
    
if __name__ == "__main__":
    sites = ['UTHSCSA', 'KUMC', 'MCW', 'UMHC', 'UNMC', 'UTSW', 'UofU', 'UPITT', 'IUR', 'UIOWA']
    
    for site in sites:
        myccmX = myccm()
        myccmX.get_onsetdata(site=site)                
        myccmX.get_data(site=site)
        myccmX.one_process_up()