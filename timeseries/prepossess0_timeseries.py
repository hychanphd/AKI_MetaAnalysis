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


class prepossess0_timeseries:
    
    def __init__(self,site):
        self.site = site
        
    def MCW_convert(self, dataname, datacols, datadtypes, sep='|', ext='dsv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()

        datacolsX = [x.lower() for x in datacols]
        datadtypesX = {key.lower(): value for key, value in datadtypes.items()}
        datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacolsX, dtype=(datadtypesX))
        datatt.columns = [x.upper() for x in datatt.columns]
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')
        return datatt    
    
    def UofU_convert(self, dataname, datacols, datadtypes, sep='|', ext='txt'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()

        datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')
        return datatt    

    def UTSW_convert(self, dataname, datacols, datadtypes, sep='|', ext='dsv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()

        if dataname == 'amed':
            sep=','
            ext='csv'
    #    datacolsX = [x.lower() for x in datacols]
    #    datadtypesX = {key.lower(): value for key, value in datadtypes.items()}
        datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
    #    datatt.columns = [x.upper() for x in datatt.columns]
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')
        return datatt    

    def UPITT_convert(self, dataname, datacols, datadtypes, sep=',', ext='csv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()
        if dataname == 'lab':
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes), encoding='windows-1252')        
        else:
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')
    #    return datatt
    #   onset = pd.read_pickle('data/'+self.site+'/p0_onset_'+self.site+'.pkl')
        print('Finished p0 '+dataname+' on self.site '+self.site, flush = True)  

    def IUR_convert(self, dataname, datacols, datadtypes, sep=',', ext='csv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()

        if dataname == 'lab':
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))        
            datatt.loc[:,'PATID']       = datatt['PATID'].map(lambda x: x.lstrip('0'))
            datatt.loc[:,'ENCOUNTERID'] = datatt['ENCOUNTERID'].map(lambda x: x.lstrip('0'))        
        else:
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')

        #    return datatt
    #   onset = pd.read_pickle('data/'+self.site+'/p0_onset_'+self.site+'.pkl')
        print('Finished p0 '+dataname+' on self.site '+self.site, flush = True)    

    def KUMC_convert(self, dataname, datacols, datadtypes, sep=',', ext='csv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()
        if dataname == 'lab':
    #        datacols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT",
            datacols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT",
                        "RESULT_NUM", "LAB_LOINC",
                        "LAB_PX_TYPE", "RESULT_UNIT", "RESULT_QUAL", "SPECIMEN_SOURCE"]
    #        datadtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
            datadtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                           "RESULT_NUM":"Float64",  "LAB_LOINC": 'object',
                           "LAB_PX_TYPE": 'object', "RESULT_UNIT": 'object', "RESULT_QUAL": 'object',
                           "SPECIMEN_SOURCE": "object"}        
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))        
            datatt = datatt[datatt['LAB_LOINC'].notnull()]
            mask =  datatt['RESULT_NUM'].isnull()
    #        datatt.loc[mask, 'RESULT_QUAL'] = datatt.loc[mask, 'RAW_RESULT']        
    #        datatt = datatt.drop('RAW_RESULT', axis=1) 
        else:
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))

        datatt = datatt.rename(columns={"ENCOUNTERID": "ENCOUNTERID"})
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')

        #    return datatt
    #   onset = pd.read_pickle('data/'+self.site+'/p0_onset_'+self.site+'.pkl')
        print('Finished p0 '+dataname+' on self.site '+self.site, flush = True)  

    def MCRI_convert(self, dataname, datacols, datadtypes, sep=',', ext='csv'):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()

        if dataname == 'amed':
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))        
            datatt = datatt[datatt['MEDADMIN_CODE'].notnull()]
        else:
            datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')

        #    return datatt
    #   onset = pd.read_pickle('data/'+self.site+'/p0_onset_'+self.site+'.pkl')
        print('Finished p0 '+dataname+' on self.site '+self.site, flush = True)  

    def read_and_save(self, dataname, datacols, datadtypes, sep=',', ext='csv'):
        print('Running p0 '+dataname+' on self.site '+self.site, flush = True)                    
        if self.site == 'UTSW':
            return self.UTSW_convert(dataname, datacols, datadtypes)
        elif self.site == 'UofU':
            return self.UofU_convert(dataname, datacols, datadtypes)
        elif self.site == 'MCW':
            return self.MCW_convert(dataname, datacols, datadtypes)
        elif self.site == 'UPITT':
            return self.UPITT_convert(dataname, datacols, datadtypes)    
        elif self.site == 'IUR':
            return self.IUR_convert(dataname, datacols, datadtypes)    
        elif self.site == 'KUMC':
            return self.KUMC_convert(dataname, datacols, datadtypes)    
        elif self.site == 'MCRI':
            return self.MCRI_convert(dataname, datacols, datadtypes)    


        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        if dataname == 'onset':
            filename = 'AKI_ONSETS'
        else:
            filename = 'AKI_'+dataname.upper()
        datatt = pd.read_csv(datafolder+self.site+'/raw/'+filename+'.'+ext,sep=sep, usecols=datacols, dtype=(datadtypes))
        datatt = datatt.rename(columns={"ENCOUNTERID": "ENCOUNTERID"})
#        datatt.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+self.site+'/p0_'+dataname+'_'+self.site+'.pkl')
        return datatt
    #   onset = pd.read_pickle('data/'+self.site+'/p0_onset_'+self.site+'.pkl')
        print('Finished p0 '+dataname+' on self.site '+self.site, flush = True)                    

    def read_and_save_onset(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # onset
        onset_cols = ['ADMIT_DATE', "PATID", "ENCOUNTERID", 
                      "NONAKI_SINCE_ADMIT", "NONAKI_ANCHOR",
                      "AKI1_SINCE_ADMIT", "AKI1_ONSET", 
                      "AKI2_SINCE_ADMIT", "AKI2_ONSET", 
                      "AKI3_SINCE_ADMIT", "AKI3_ONSET"]
        onset_dtypes =  {'ADMIT_DATE': 'object', "PATID": 'object', "ENCOUNTERID": 'object', 
                         "NONAKI_SINCE_ADMIT": 'Int64', "NONAKI_ANCHOR": 'object',                      
                         "AKI1_SINCE_ADMIT": 'Int64', "AKI1_ONSET": 'object', 
                         "AKI2_SINCE_ADMIT": 'Int64', "AKI2_ONSET": 'object', 
                         "AKI3_SINCE_ADMIT": 'Int64', "AKI3_ONSET": 'object'}
        return self.read_and_save('onset', onset_cols, onset_dtypes)    

    def read_and_save_vital(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # vital
        vital_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT", 
                      "SYSTOLIC", "DIASTOLIC", "ORIGINAL_BMI", "WT"]
        vital_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                         "SYSTOLIC": 'Float64', "DIASTOLIC": 'Float64', "ORIGINAL_BMI": 'Float64', "WT": 'Float64'}
        return self.read_and_save('vital', vital_cols, vital_dtypes)    

    def read_and_save_demo(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        #demo
        demo_cols = ["PATID", "ENCOUNTERID", 
                      "AGE", "SEX", "RACE", "HISPANIC"]
        demo_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', 
                         "SEX": 'Int64', "SEX": 'category', "RACE": 'category', "HISPANIC": 'category'}
        return self.read_and_save('demo', demo_cols, demo_dtypes)    

    def read_and_save_dx(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # dx
        dx_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT", 
                      "DX", "DX_TYPE"]
        dx_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                         "DX": 'object', "DX_TYPE": 'object'}
        return self.read_and_save('dx', dx_cols, dx_dtypes)

    def read_and_save_px(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # px
        px_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT", 
                      "PX", "PX_TYPE"]
        px_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                         "PX": 'object', "PX_TYPE": 'object'}
        return self.read_and_save('px', px_cols, px_dtypes)    

    def read_and_save_lab(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # lab
    #    lab_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT",
        lab_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT",
                    "RESULT_NUM", "LAB_LOINC",
                    "LAB_PX_TYPE", "RESULT_UNIT", "RESULT_QUAL", "SPECIMEN_SOURCE"]
        lab_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64',     
    #    lab_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                       "RESULT_NUM":"Float64",  "LAB_LOINC": 'object',
                       "LAB_PX_TYPE": 'object', "RESULT_UNIT": 'object', "RESULT_QUAL": 'object', "SPECIMEN_SOURCE": 'object'}
        read_and_save('lab', lab_cols, lab_dtypes)    

    def read_and_save_lab_all(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        read_and_save('laball', None, None)    

    def read_and_save_amed(self):
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'

        # amed
        amed_cols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT", 
                      "MEDADMIN_TYPE", "MEDADMIN_CODE", "MEDADMIN_ROUTE", "MEDADMIN_START_DATE_TIME"]
        amed_dtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                         "MEDADMIN_TYPE": "object", "MEDADMIN_CODE": 'object', "MEDADMIN_ROUTE": 'object', "MEDADMIN_START_DATE_TIME": 'object'}

        return self.read_and_save('amed', amed_cols, amed_dtypes)    

    def read_and_save_all(self):
        self.read_and_save_onset(site)
        self.read_and_save_demo(site)
        self.read_and_save_vital(site)
        self.read_and_save_dx(site)
        self.read_and_save_px(site)
        self.read_and_save_lab(site)
        self.read_and_save_amed(site)
    