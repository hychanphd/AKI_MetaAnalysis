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


def runner(runner_wrapper, site, year=None, stg=None, fs=None, oversample=None, model_type=None, numberbt=None):
    
    #global setting (no ideal)
    datafolder = '/home/hchan2/AKI/data/'
    home_directory = "/home/hchan2/AKI/AKI_Python/"
    pred_end = 7    

    tic = time.perf_counter() 
#    if type(site) is list:
#        runner_wrapper(site[0], site[1], year, stg, fs, oversample, model_type, numberbt)
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
    print(f"{site['site']}:{year} finished in {toc - tic:0.4f} seconds", flush=True)        
    
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

sites = ['UTHSCSA', 'UTSW', 'MCW', 'UofU', 'UIOWA', 'UMHC', 'UNMC', 'KUMC', 'UPITT']
#sites = ['MCRI', 'IUR', 'MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UPITT', 'UTHSCSA', 'KUMC', 'UTSW']

#sites = ['UTHSCSA']

#sites = ['UTSW', 'UMHC', 'UPITT', 'KUMC']
#sites = ['UPITT']

configs_variables = [utils_function.read_config(site) for site in sites]

# Re-run?
for configs_variable in configs_variables:
    configs_variable['rerun_flag'] = True

n_splits = int(configs_variables[0]['n_splits'])

#runner_wrapper_list2 = [runxgboost.boosttrapcatboost]
#parasites2 = [(runner, configs_variable, i, None, None, None, None, None) for configs_variable in configs_variables for runner in runner_wrapper_list2 for i in range(n_splits)]

runner_wrapper_list2 = [postprocessing1_SHAP.collectSHAP_cross_sub_validate]
parasites2 = [(runner, configs_variable_m, configs_variable_d, i, None, None, None, None) for configs_variable_m in configs_variables for configs_variable_d in configs_variables for runner in runner_wrapper_list2 for i in range(n_splits)]

runner_wrapper_list3 = [postprocessing1_SHAP.collectSHAPraw_cross_sub]
parasites3 = [(runner, configs_variable_m, configs_variable_d, None, None, None, None) for configs_variable_m in configs_variables for configs_variable_d in configs_variables for runner in runner_wrapper_list3]

# Metaanlysis Data Collection
runner_wrapper_list_13 = [ipynb.fs.full.postprocessing3_collect.collect_collectSHAPraw_cross_sub_pre]
parasites_13 = [(runner, configs_variables, None, None, None, None, None, None) in configs_variables for runner in runner_wrapper_list_13]

patasites_meta = [parasites_13]

#patasites_meta = [parasites2, parasites3]

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
    with ProcessPool(max_workers=34) as pool:
        for paras in parasites:
            future = pool.schedule(runner, args=paras, timeout=86400)
            future.add_done_callback(task_done)
        
print('done', flush=True)
ping_slack('done:AKI_Python:runner.py')