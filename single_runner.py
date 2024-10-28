import importlib

import ipynb.fs.full.preprocessing0
import ipynb.fs.full.preprocessing05
#import ipynb.fs.full.prepossessing075_akistage
import ipynb.fs.full.preprocessing1
import ipynb.fs.full.preprocessing2_BT
import ipynb.fs.full.preprocessing25_BTcorr
import ipynb.fs.full.preprocessing3_smote
#import ipynb.fs.full.preprocessing4
import preprocessing4

import ipynb.fs.full.runxgboost
import ipynb.fs.full.postprocessing1_SHAP
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

def runner(runner_wrapper, site, year=None, stg=None, fs=None, oversample=None, model_type=None, numberbt=None):
    
    #global setting (no ideal)
    datafolder = '/home/hchan2/AKI/data/'
    home_directory = "/home/hchan2/AKI/AKI_Python/"
    pred_end = 7    
    
#    filename = 'data/'+site+'/model_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
#    if exists(filename):
#        print(filename +" already exists")
#       return

    tic = time.perf_counter() 
    if not numberbt is None:
        runner_wrapper(site, year, stg, fs, oversample, model_type, numberbt)                
    if not fs is None:
        runner_wrapper(site, year, stg, fs, oversample, model_type)        
    elif not stg is None:
        runner_wrapper(site, year, stg)        
    elif not year is None:
        runner_wrapper(site, year)
    else:
        runner_wrapper(site)        
    toc = time.perf_counter()
    print(f"{site}:{year} finished in {toc - tic:0.4f} seconds")        

importlib.reload(ipynb.fs.full.preprocessing0)
importlib.reload(ipynb.fs.full.preprocessing05)
#importlib.reload(ipynb.fs.full.prepossessing075_akistage)
importlib.reload(ipynb.fs.full.preprocessing1)
importlib.reload(ipynb.fs.full.preprocessing2_BT)
importlib.reload(ipynb.fs.full.preprocessing25_BTcorr)
importlib.reload(ipynb.fs.full.preprocessing3_smote)
#importlib.reload(ipynb.fs.full.preprocessing4)
importlib.reload(preprocessing4)
importlib.reload(ipynb.fs.full.runxgboost)
importlib.reload(ipynb.fs.full.postprocessing1_SHAP)
importlib.reload(ipynb.fs.full.postprocessing3_collect)

#sites = ['MCRI', 'IUR', 'MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UPITT', 'UTHSCSA', 'KUMC', 'UTSW']
#sites = ['MCRI', 'MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UPITT', 'UTHSCSA', 'UTSW']
#sites = ['MCRI', 'MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UTHSCSA', 'UTSW']
#sites = ['MCRI', 'UMHC', 'UNMC', 'KUMC']
sites = ['IUR']

#sites = ['UPITT']
#sites=['KUMC']
#problematic_sites = ['IUR']

#runner_wrapper_list = [ipynb.fs.full.preprocessing0.read_and_save_lab]

#runner_wrapper_list = [ipynb.fs.full.preprocessing0.read_and_save_onset]
# runner_wrapper_list = [ipynb.fs.full.preprocessing0.read_and_save_demo,
#                        ipynb.fs.full.preprocessing0.read_and_save_vital,
#                        ipynb.fs.full.preprocessing0.read_and_save_dx,
#                        ipynb.fs.full.preprocessing0.read_and_save_px,
#                        ipynb.fs.full.preprocessing0.read_and_save_amed,                      
#                        ipynb.fs.full.preprocessing0.read_and_save_lab]
runner_wrapper_list = [ipynb.fs.full.preprocessing1.unify_lab]

#runner_wrapper_list = [ipynb.fs.full.preprocessing05.lab_drop_outliner]
#runner_wrapper_list = [ipynb.fs.full.preprocessing05.vital_drop_outliner]

#runner_wrapper_list = [ipynb.fs.full.prepossessing075_akistage.aki_staging]
#runner_wrapper_list = [ipynb.fs.full.prepossessing075_akistage.generate_ckd_table]


#runner_wrapper_list = [ipynb.fs.full.preprocessing1.lab]
# runner_wrapper_list = [ipynb.fs.full.preprocessing1.demo,
#                          ipynb.fs.full.preprocessing1.vital,
#                          ipynb.fs.full.preprocessing1.dx,
#                          ipynb.fs.full.preprocessing1.px,
#                          ipynb.fs.full.preprocessing1.lab,                                              
#                          ipynb.fs.full.preprocessing1.amed]                        

#runner_wrapper_list = [ipynb.fs.full.preprocessing2_BT.bigtable]
#runner_wrapper_list = [ipynb.fs.full.preprocessing25_BTcorr.generate_corr]
#runner_wrapper_list = [ipynb.fs.full.preprocessing25_BTcorr.generate_corr_bt]

#runner_wrapper_list = [preprocessing4.combinebtpos]
#runner_wrapper_list = [preprocessing4.combinebt]
#runner_wrapper_list = [ipynb.fs.full.preprocessing4.combinebtpos]
#runner_wrapper_list = [ipynb.fs.full.preprocessing4.combinebt]

#runner_wrapper_list = [ipynb.fs.full.preprocessing3_smote.pre_smote]
#runner_wrapper_list = [ipynb.fs.full.preprocessing3_smote.smote]
#runner_wrapper_list = [ipynb.fs.full.preprocessing3_smote.generate_all_pre_catboost]

#runner_wrapper_list = [ipynb.fs.full.runxgboost.runxgboost]
#runner_wrapper_list = [ipynb.fs.full.postprocessing1_SHAP.collectSHAP_sub]

#runner_wrapper_list = [ipynb.fs.full.runxgboost.boosttrapcatboost]

parasites = [(runner, site, None, None, None, None, None, None) for site in sites for runner in runner_wrapper_list]