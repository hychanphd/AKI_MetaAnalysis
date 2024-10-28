'''
main predictive algorithm
'''

import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from skopt import BayesSearchCV
from catboost import Pool, cv

import utils_function
        
import preprocessing4
import importlib
import ipynb.fs.full.preprocessing3_smote
importlib.reload(ipynb.fs.full.preprocessing3_smote)

def xgbHalvingGridSearchCV(X_train, y_train):
    labelcount = y_train.value_counts()
    params = {
            'subsample': [0.5, 1], #subsample data set with grow tree
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1, 0.5],
            'gamma' :[0.5, 1]        
            }

    #scale_pos_weight for imbalanced data
    cvmodel = XGBClassifier(n_jobs=1, scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='binary:logistic', eval_metric='auc', verbosity=0, 
                            early_stopping_rounds=50, use_label_encoder=False)

    # skf = StratifiedKFold(n_splits=5)
    # random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=n_trial, scoring='roc_auc', n_jobs=n_trial, cv=skf.split(X_train_onehot_com,y_train_com), verbose=3, random_state=1001)
    # random_search.fit(X_train_onehot_com, y_train_com)

    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV

    #search_obj = HalvingGridSearchCV(cvmodel, params, verbose=3, n_jobs=23)
    search_obj = HalvingGridSearchCV(cvmodel, params, verbose=3, n_jobs=20, resource='n_estimators', max_resources=2000, min_resources=100, aggressive_elimination=True)
    search_result = search_obj.fit(X_train, y_train)
    bestmodel = search_result.best_estimator_
    bestmodel.set_params(n_jobs=23)
    return bestmodel

def xgbBayesSearchCV(X_train, y_train):
    labelcount = y_train.value_counts()
    cvmodel = XGBClassifier(n_jobs=4, scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='binary:logistic', eval_metric='auc', verbosity=0, 
    #                        early_stopping_rounds=50, n_estimators=1000, use_label_encoder=False)
                            early_stopping_rounds=50, use_label_encoder=False)

    params = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),    
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),    
        'gamma': (1e-9, 1.0, 'log-uniform'),
        'n_estimators': (50, 1000),
    }
    skf = StratifiedKFold(n_splits=5)
    bayes_cv_tuner = BayesSearchCV(estimator=cvmodel, search_spaces=params, cv=skf, n_jobs=5, verbose=3, refit = True, n_iter=50)
    bayes_cv_tuner.fit(X_train, y_train)
    bestmodel = bayes_cv_tuner.best_estimator_
    bestmodel.set_params(n_jobs=23)
    return bestmodel

def catDefault(X_train, y_train):
    labelcount = y_train.value_counts()    
    cat_features = list(X_train.select_dtypes('bool').columns)
    cvmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='Logloss', eval_metric='AUC', verbose=50,
                            early_stopping_rounds=50, cat_features=cat_features,                                 
                            custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
    return cvmodel

def catRandomSearch(X_train, y_train):
    labelcount = y_train.value_counts()    
    cat_features = list(X_train.select_dtypes('bool').columns)    
    cvmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='Logloss', eval_metric='AUC:hints=skip_train~false', verbose=50, 
                            early_stopping_rounds=50, cat_features=cat_features)
    params = {
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bylevel': [0.1, 0.5, 1.0],
            'max_depth': [5, 7, 16],
            'learning_rate': [0.1, 0.5],
            'n_estimators': [50, 200, 1000]
            }
    randomized_search_result = cvmodel.randomized_search(params, X=X_train, y=y_train, cv=5, n_iter=20)
    bestmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                               objective='Logloss', eval_metric='AUC:hints=skip_train~false', verbose=50, 
                                early_stopping_rounds=50, cat_features=cat_features, **randomized_search_result['params'])
    return bestmodel

def runxgboost(configs_variables, returnflag=False, X_train=None, X_test=None, y_train=None, y_test=None):
    
    year=3000
#    configs_variables = utils_function.read_config(site)
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    stg = configs_variables['stg']
    fs = configs_variables['fs']
    oversample = configs_variables['oversample']
    model_type = configs_variables['model_type']
    drop_correlation_catboost = configs_variables['drop_correlation_catboost']
        
    print('Running '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    
    if drop_correlation_catboost:
        suffix='nc'
    else:
        suffix= ''
    
    #load tables
    if X_train is None:
        X_train = pd.read_pickle(datafolder+site+'/X_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        X_test =  pd.read_pickle(datafolder+site+ '/X_test_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_train = pd.read_pickle(datafolder+site+'/y_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_test =  pd.read_pickle(datafolder+site+ '/y_test_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train = X_train.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test = X_test.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
        
        
    tic = time.perf_counter()     
    #xgboost
    if model_type == "xgbhgs":
        bestmodel = xgbHalvingGridSearchCV(X_train, y_train)
        bestmodel.set_params(n_jobs=23)
    if model_type == "xgbbs":        
        bestmodel = xgbBayesSearchCV(X_train, y_train)
        bestmodel.set_params(n_jobs=23)

    #catboost
    if model_type == "catd":
        bestmodel = catDefault(X_train, y_train)
    if model_type == "catr":
        bestmodel = catRandomSearch(X_train, y_train)

    print('Training xgb/cat on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    bestmodel.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50, early_stopping_rounds=50)
    prelabel = bestmodel.predict(X_test)

    pred = bestmodel.predict_proba(X_test)
    roc = roc_auc_score(y_test, pred[:,1])    
    
    print('roc = '+ str(roc))
    print('Confusion Matrix')
    cm = confusion_matrix(y_test, prelabel)
    print(cm)
    
    toc = time.perf_counter()
    print('Finished '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)        
    print(f"{site}:{year}:{stg}:{fs}:{oversample}: finished in {toc - tic:0.4f} seconds")  
    if returnflag:
        return bestmodel, roc, cm
    pickle.dump(bestmodel, open(datafolder+site+'/model_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'wb'))    
    
#    pickle.dump(bestmodel, open(datafolder+site+'/model_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'wb'))        
    
def boosttrapcatboost(configs_variables, numberbt):      

    '''
    This module run on cross validation dataset
    '''    
    year = 3000
#    configs_variables = utils_function.read_config(site)
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    fs = configs_variables['fs']
    stg = configs_variables['stg']
    oversample = configs_variables['oversample']
    model_type = configs_variables['model_type']
    
    drop_correlation_catboost = configs_variables['drop_correlation_catboost']    
    n_splits = int(configs_variables['n_splits'])
    random_state = int(configs_variables['random_state'])
    
    if drop_correlation_catboost:
        suffix = 'nc'
    else:
        suffix = ''      
    
    print('Training BT ' +str(numberbt)+ ' cat on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    
    X_train, X_test, y_train, y_test = ipynb.fs.full.preprocessing3_smote.get_boosttrap(configs_variables, numberbt)
    
    bestmodel, roc, cm = runxgboost(configs_variables, returnflag=True, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)       
    
    saveobjpkl = (configs_variables, numberbt, bestmodel, roc, cm)
    pickle.dump(saveobjpkl, open(datafolder+site+'/boosttrap_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(numberbt)+'.pkl', 'wb'))    
    
    print(datafolder+site+'/boosttrap_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(numberbt)+'.pkl')
    