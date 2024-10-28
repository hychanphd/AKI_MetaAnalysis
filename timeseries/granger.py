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

import ipynb.fs.full.preprocessing1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests


class granger:
    def __init__(self):
        pass
    
    def regular_ts(self, s):
        oidx = s.index
        nidx = pd.date_range(oidx.min(), oidx.max(), freq='6h')
        res = s.reindex(oidx.union(nidx)).interpolate('index').reindex(nidx)
        res.plot(style='.-')
        s.plot(style='o')    
        return res
    
    def lag_plots(self, data_df):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        lag_plot(data_df[data_df.columns[0]], ax=ax1)
        ax1.set_title(data_df.columns[0]);

        lag_plot(data_df[data_df.columns[1]], ax=ax2)
        ax2.set_title(data_df.columns[1]);

        ax1.set_ylabel('$y_{t+1}$');
        ax1.set_xlabel('$y_t$');
        ax2.set_ylabel('$y_{t+1}$');
        ax2.set_xlabel('$y_t$');

        plt.tight_layout()
    
    def kpss_test(self, data_df):
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
        return kpss_res_df
    
    def adf_test(self, data_df):
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
        return adf_res_df
    
    def stationarize(self, df):
        return df.diff().dropna()

    def splitter(self, data_df):
        end = round(len(data_df)*.8)
        train_df = data_df[:end]
        test_df = data_df[end:]
        return train_df, test_df
    
    def select_p(self, train_df):
        aic, bic, fpe, hqic = [], [], [], []
        model = VAR(train_df) 
        p = np.arange(1,15)
        for i in p:
            result = model.fit(i)
            aic.append(result.aic)
            bic.append(result.bic)
            fpe.append(result.fpe)
            hqic.append(result.hqic)
        lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                        'BIC': bic, 
                                        'HQIC': hqic,
                                        'FPE': fpe}, 
                                       index=p)    
        fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharex=True)
        lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
        plt.tight_layout()
        print(lags_metrics_df.idxmin(axis=0))
        
    def run_VAR(self, train_df, p=10):
        p = 10
        model = VAR(train_df)
        var_model = model.fit(maxlags=p)
        
    def granger_causation_matrix(self, data, variables, p, test = 'ssr_chi2test', verbose=False):    
        """Check Granger Causality of all possible combinations of the time series.
        The rows are the response variables, columns are predictors. The values in the table 
        are the P-Values. P-Values lesser than the significance level (0.05), implies 
        the Null Hypothesis that the coefficients of the corresponding past values is 
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], p, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(p)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        return df
    
    def plot_irf(self, var_model, ax = None, period=50, cum=True):
        if ax == None:
            fig, ax = plt.figure()
        irf = var_model.irf(periods=50)
        if cum:
            ax = irf.plot_cum_effects(orth=False, subplot_params={'fontsize': 10})
        else:
            ax = irf.plot(orth=False, 
                  subplot_params={'fontsize': 10})

    def print_summary(self, var_model):
        var_model.summary()