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

from datetime import datetime, timedelta
import utils_function

import parquet_splitter

def runner(runner_wrapper, site, year=None, stg=None, fs=None, oversample=None, model_type=None, numberbt=None):
    
    #global setting (no ideal)
    datafolder = '/home/hchan2/AKI/data/'
    home_directory = "/home/hchan2/AKI/AKI_Python/"
    pred_end = 7    

    tic = time.perf_counter() 
    if not numberbt is None:        
        runner_wrapper(site, year, stg, fs, oversample, model_type, numberbt)                
    elif not fs is None:
        runner_wrapper(site, year, stg, fs, oversample, model_type)        
    elif not stg is None:
        runner_wrapper(site, year, stg)        
    elif not year is None:
        runner_wrapper(site, year)
    else:
        runner_wrapper(site)        
    toc = time.perf_counter()
    
    if len(site) > 10:
        print(f"{site['site']}:{year} finished in {toc - tic:0.4f} seconds", flush=True)        
    else:
        print(f"All sites finished in {toc - tic:0.4f} seconds", flush=True)                
    
def parasites_gen_year(runner_wrapper_list, configs_variables):
    parasites = []
    for configs_variable in configs_variables:
        onset = pd.read_parquet(configs_variable['datafolder']+configs_variable['site']+'/p0_onset_'+configs_variable['site']+'.parquet')
        years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
        para_list_local = [(runner, configs_variable, year, None, None, None, None, None) for year in years for runner in runner_wrapper_list]
        parasites.extend(para_list_local)    
    return parasites
        
importlib.reload(ipynb.fs.full.preprocessing0)
importlib.reload(ipynb.fs.full.preprocessing05)
#importlib.reload(ipynb.fs.full.prepossessing075_akistage)
importlib.reload(preprocessing1)
importlib.reload(preprocessing2_BT)
importlib.reload(ipynb.fs.full.preprocessing25_BTcorr)
importlib.reload(ipynb.fs.full.preprocessing3_smote)
# #importlib.reload(ipynb.fs.full.preprocessing4)
importlib.reload(preprocessing4)
importlib.reload(runxgboost)
importlib.reload(postprocessing1_SHAP)
importlib.reload(ipynb.fs.full.postprocessing3_collect)
importlib.reload(postprocessing1_SHAP)
importlib.reload(utils_function)
importlib.reload(parquet_splitter)

#datanames = ['demo', 'vital_old', 'vital_old_nooutliner', 'dx', 'px', 'lab_g', 'lab_g_nooutliner', 'amed']
datanames = ['lab_g', 'lab_g_nooutliner']

sites = ['UTHSCSA', 'UTSW', 'MCW', 'UofU', 'UIOWA', 'UMHC', 'UNMC', 'KUMC', 'UPITT']
#sites = ['KUMC', 'UPITT']


configs_variables = [utils_function.read_config(site) for site in sites]

# Re-run?
for configs_variable in configs_variables:
    configs_variable['rerun_flag'] = True


    
n_splits = int(configs_variables[0]['n_splits'])

#pre-pre-processing
runner_wrapper_list_n1 = [preprocessing1.unify_lab]
parasites_n1 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_n1]

runner_wrapper_list_n2 = [ipynb.fs.full.preprocessing05.lab_drop_outliner,
                          ipynb.fs.full.preprocessing05.vital_drop_outliner]
parasites_n2 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_n2]


runner_wrapper_list_n3 = [parquet_splitter.spliter]
parasites_n3 = [(runner, configs_variable['site'], dataname, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_n3 for dataname in datanames]


#preprocessing
runner_wrapper_list_0 = [preprocessing1.onset]
parasites_0 = parasites_gen_year(runner_wrapper_list_0, configs_variables)

runner_wrapper_list_1 = [preprocessing1.demo,
                        preprocessing1.vital,
                        preprocessing1.dx,
                        preprocessing1.px,
                        preprocessing1.lab,                                              
                        preprocessing1.amed] 
#parasites_1 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_1]
parasites_1 = parasites_gen_year(runner_wrapper_list_1, configs_variables)

runner_wrapper_list_2 = [preprocessing2_BT.bigtable]
#parasites_2 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_2]
parasites_2 = parasites_gen_year(runner_wrapper_list_2, configs_variables)

runner_wrapper_list_3 = [preprocessing4.combinebtpos]
parasites_3 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_3]


runner_wrapper_list_35 = [utils_function.get_bool_columns]
parasites_35 = [(runner, configs_variables, None, None, None, None, None, None) for runner in runner_wrapper_list_35]


#Remove correlation
runner_wrapper_list_4 = [ipynb.fs.full.preprocessing25_BTcorr.generate_corr]
parasites_4 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_4]

runner_wrapper_list_5 = [ipynb.fs.full.preprocessing25_BTcorr.calculate_corr_occurence_new]
parasites_5 = [(runner, configs_variables, None, None, None, None, None, None) for runner in runner_wrapper_list_5]

runner_wrapper_list_6 = [ipynb.fs.full.preprocessing25_BTcorr.remove_correlated_features]
parasites_6 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_6]



# Meta-analysis Data
runner_wrapper_list_7 = [ipynb.fs.full.preprocessing3_smote.generate_all_pre_catboost]
parasites_7 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_7]

runner_wrapper_list_8 = [runxgboost.runxgboost]
parasites_8 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_8]

runner_wrapper_list_9 = [postprocessing1_SHAP.collectSHAPraw_cross_sub]
parasites_9 = [(runner, configs_variable_m, configs_variable_d, None, None, None, None) for configs_variable_m in configs_variables for configs_variable_d in configs_variables for runner in runner_wrapper_list_9]

#10-fold cross validation
runner_wrapper_list_10 = [ipynb.fs.full.preprocessing3_smote.gen_crossvalidate]
parasites_10 = [(runner, configs_variable, None, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_10]

runner_wrapper_list_11 = [runxgboost.boosttrapcatboost]
parasites_11 = [(runner, configs_variable, i, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list_11 for i in range(n_splits)]

runner_wrapper_list_12 = [postprocessing1_SHAP.collectSHAP_cross_sub_validate]
parasites_12 = [(runner, configs_variable_m, configs_variable_d, i, None, None, None, None) for configs_variable_m in configs_variables for configs_variable_d in configs_variables for runner in runner_wrapper_list_12 for i in range(n_splits)]

# Meta-analysis Data Collection
runner_wrapper_list_13 = [ipynb.fs.full.postprocessing3_collect.collect_collectSHAPraw_cross_sub_pre]
parasites_13 = [(runner, configs_variables, None, None, None, None, None, None) for runner in runner_wrapper_list_13]

# runner_wrapper_list_14 = [ipynb.fs.full.postprocessing3_collect.collect_collectSHAPraw_cross_sub]
# parasites_14 = [(runner, configs_variables, None, None, None, None, None, None)]


# Parallel Runner
patasites_meta = [parasites_0, parasites_1, parasites_2, parasites_3, parasites_35, parasites_4, parasites_5, parasites_6, parasites_7, parasites_8, parasites_9, parasites_10, parasites_11, parasites_12, parasites_13]
#patasites_meta = [parasites_5, parasites_6, parasites_7, parasites_8, parasites_9, parasites_10, parasites_11, parasites_12, parasites_13]
#patasites_meta = [parasites_0, parasites_1, parasites_2, parasites_3, parasites_35, parasites_4]
patasites_meta = [parasites_12]
#patasites_meta = [parasites_n1, parasites_n2] #max_worker=2
#patasites_meta = [parasites_n3] #max_worker=2

import multiprocessing as mp
from pebble import ProcessPool
from concurrent.futures import TimeoutError
def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1], flush=True)
    except Exception as error:
        print("Function raised " + "error" +"\n"+error.traceback, flush=True)

for parasites in patasites_meta:  
    tic_meta = time.perf_counter() 
    print(f"Meta-Running {parasites[0][0]} Start", flush=True)
    with ProcessPool(max_workers=20) as pool:
#    with ProcessPool(max_workers=2) as pool:
        for paras in parasites:
            future = pool.schedule(runner, args=paras, timeout=86400)
            future.add_done_callback(task_done)
    toc_meta = time.perf_counter()            
    print(f"Meta-Running {parasites[0][0]} finished in {toc_meta - tic_meta:0.4f} seconds", flush=True)
    
print('done', flush=True)
ping_slack('done:AKI_Python:runner.py')


