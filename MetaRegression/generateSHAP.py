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
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
import csv
from dfply import *
from xgboost import XGBClassifier
import itertools
import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import pickle
import math
import matplotlib.pyplot as plt
from glob import glob

import importlib

import time
import requests

from joblib import Parallel, delayed
from joblib import parallel_backend
import seaborn as sns

from catboost import Pool, cv
import shap as sp
from os.path import exists
import itertools

class generateSHAP:
    
    def __init__(self):
        self.datafolder = '/home/hoyinchan/blue/Data/data2021/data2021/'
        self.home_directory = "/home/hoyinchan/code/AKI_CDM_PY/"
        self.sites = ['MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UTHSCSA', 'UTSW', 'IUR', 'KUMC', 'UPITT']
        self.stgs = ["stg01"]
        self.fss =  ['nofs']
        self.oversamples = ['raw']
        self.model_types = ['catd']
        self.barX = None
        self.shap = None
        
    def load_test_data(self, site_m, site_d, year, stg, fs, oversample, model_type):
        #load tables
#        X_train_m = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
#        y_train_m = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
        X_test_m = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
#        y_test_m = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')

        X_train_d = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_d+'/X_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
        y_train_d = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_d+'/y_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
        X_test_d =  pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')
        y_test_d =  pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'pos.pkl')

        X_test3_d, y_test_d  = self.feature_match(X_test_d, X_test_m, y_test_d)
        X_train3_d, y_train_d = self.feature_match(X_train_d, X_test_m, y_train_d)
        
        return [X_train3_d, X_test3_d, y_train_d, y_test_d]
    
    def cross_roc(self, site_m, site_d, year, stg, fs, oversample, model_type, ckd_group=0, returnflag=False, rerun=False):
#        try:
        filepath = self.home_directory+'cross_roc_tmp/'+'crossroctmp_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
        if rerun or not exists(filepath):
            print('Running cross_roc '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
            model = pickle.load(open('/home/hoyinchan/blue/Data/data2021/data2021/'+
                                     site_m+'/model_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))

            [X_train3_d, X_test3_d, y_train_d, y_test_d] = self.load_test_data(site_m, site_d, year, stg, fs, oversample, model_type)

            pred = model.predict_proba(X_test3_d)
            roc = roc_auc_score(y_test_d, pred[:,1])
            print('Finished cross_roc '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
            pd.DataFrame([site_m, site_d, stg, fs, oversample, model_type, roc]).to_pickle(filepath)
#            return [site_m, site_d, stg, fs, oversample, model_type, roc]
#        except:
#            pass
#            if rerun or not exists(filepath):
#                pd.DataFrame([site_m, site_d, stg, fs, oversample, model_type, np.nan]).to_pickle(filepath)
    #            return [site_m, site_d, stg, fs, oversample, model_type, np.nan]     

    def read_tmp_cross_roc(site_m, site_d, year, stg, fs, oversample, model_type, ckd_group=0):
        try:
            filepath = self.home_directory+'cross_roc_tmp/'+'crossroctmp_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
            return pd.read_pickle(filepath)
        except:
            return pd.DataFrame([site_m, site_d, stg, fs, oversample, model_type, np.nan])    
    
    def feature_match(self, data_d, data_m, data_yd):
        common_features = [x for x in data_d.columns if x in data_m.columns]
        data2_d = data_d[common_features].copy()
        data2_m = data_m.iloc[0:1].copy()
#        data3_d = pd.concat([data2_m, data2_d]).iloc[1:].copy()
#        data3_d.loc[:,data2_m.dtypes==bool] = data3_d.loc[:,data2_m.dtypes==bool].fillna(False)
        
        def df_add_column(bt, bool_feature, allcols):
            new_bool = [x for x in bool_feature if x not in bt.columns]
            new_bool_df = bt.reindex(columns=new_bool, fill_value=False)
            bt = pd.concat([bt.T,new_bool_df.T]).T
            bt = bt.reindex(columns=allcols)
            return bt        
        
        bool_feature = data2_m.select_dtypes(bool).columns
        data3_d = df_add_column(data2_d, bool_feature, data2_m.columns)            
        
        data_yd = data_yd[data3_d['AGE'].notnull()]        
        data3_d = data3_d[data3_d['AGE'].notnull()]
        data3_d = data3_d.astype(data_m.dtypes.to_dict())            
            
        # for i in np.where(data3_d.dtypes != data_m.dtypes)[0]:
        #     if data_m.iloc[:,i].dtypes == bool:
        #         data3_d.iloc[:,i] = False
        #     else:
        #         data3_d.iloc[:,i] = np.nan
        data3_d = data3_d[data_m.columns]        
        
        return data3_d, data_yd
    
    def cross_roc_all(self, n_jobs=1):
        Parallel(n_jobs=n_jobs)(delayed(self.cross_roc)(site_m, site_d, 3000, stg, fs, oversample, model_type) 
                                     for site_m in self.sites for site_d in self.sites for stg in self.stgs 
                                     for fs in self.fss for oversample in self.oversamples for model_type in self.model_types)   
        
    def save_cross_roc(self):
        with open(self.datafolder+"tmp_crossroc.pkl", "wb") as f:
            pickle.dump(self.barX, f)
            
    def load_cross_roc(self):
        with open(self.datafolder+"tmp_crossroc.pkl", "rb") as f:
            self.barX = pickle.load(f)
    
    def roc_to_table(self):
#        barXX = pd.DataFrame(self.barX)
        def read_tmp_cross_roc_sub(site_m, site_d, year, stg, fs, oversample, model_type, ckd_group=0):
            try:
                filepath = self.home_directory+'cross_roc_tmp/'+'crossroctmp_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
                return pd.read_pickle(filepath).T
            except:
                return pd.DataFrame([site_m, site_d, stg, fs, oversample, model_type, np.nan]).T
        barXX = pd.concat([read_tmp_cross_roc_sub(site_m, site_d, 3000, stg, fs, oversample, model_type, ckd_group=0) for site_m in self.sites for site_d in self.sites for stg in self.stgs for fs in self.fss for oversample in self.oversamples for model_type in self.model_types]) 
        barXX.columns = ['site_m', 'site_d', 'stg', 'fs', 'oversample', 'model_type', 'roc']
        barXX = barXX.reset_index()
        return barXX
    
    def roc_heat_map(self, save=False):
        barXX = self.roc_to_table()
        barX = barXX[np.logical_and(barXX['stg']=='stg01',barXX['fs']=='nofs')]
        roc_table = np.round(barX[['site_m', 'site_d', 'roc']].pivot(index='site_m', columns='site_d', values='roc'),2)
        roc_table = np.round(roc_table.astype(float),2)
        
        plt.subplots(figsize=(10,8.5))
        sns.heatmap(roc_table, cmap='coolwarm', annot=roc_table, fmt = '', linewidths=0.01, linecolor='white')
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
        if save:
            plt.savefig('roc_table.svg', format='svg', bbox_inches='tight')
        
    def copy(self, otherobj):
        self.barX = otherobj.barX
        self.feature_importance = otherobj.feature_importance
    
    def gen_samples(self, shapX, shapy, number_of_sample, sample_x=False):
        if sample_x and shapX.shape[0] > number_of_sample:
            shap_sam = pd.concat([shapX, shapy],axis=1)
            fract = number_of_sample/shap_sam.shape[0]
            shap_sam = shap_sam.groupby('FLAG').apply(lambda x: x.sample(frac=fract))
            shap_sam = shap_sam.droplevel('FLAG')
            shapX = shap_sam[[x for x in shap_sam.columns if x != 'FLAG']]
            shapy = shap_sam[['FLAG']]
        return shapX, shapy
    
    def gen_cat_sample(self, shapX, shapy, number_of_sample, cat_feature):
        shap_sam = pd.concat([shapX, shapy],axis=1)
        fract = number_of_sample/shap_sam.shape[0]
        number_of_sample_f = np.floor(shapX[cat_feature].sum()/shapX.shape[0]*number_of_sample).astype(int)        
        
        shapX_list = list()
        shapy_list = list()
        for f in cat_feature:
            shap_samf = shap_sam[shap_sam[f]]
            if shap_samf.shape[0] > number_of_sample_f[f] and shap_samf.shape[0] != 0 and shap_sam.shape[0] < number_of_sample:
                fract_f = number_of_sample_f[f]/shap_samf.shape[0]
                shap_samf = shap_samf.groupby('FLAG').apply(lambda x: x.sample(frac=fract))
                shap_samf = shap_samf.droplevel('FLAG')
            shapXf = shap_samf[[x for x in shap_samf.columns if x != 'FLAG']]
            shapyf = shap_samf[['FLAG']]
            shapX_list.append(shapXf)
            shapy_list.append(shapyf)
        shapX = pd.concat(shapX_list)
        shapy = pd.concat(shapy_list)        
        return shapX, shapy
    
    def getshap(self, site_m, site_d, year, stg, fs, oversample, model_type, n_jobs=1, save=False, sample_x=False, number_of_sample=50000, sample_y=False, cat_feature_flag=False):
        try:
            if cat_feature_flag:
                print('cat_mode')
            print('Running getshap '+model_type+' on site '+site_m+'/'+site_d+":"+str(3000)+":"+stg+":"+fs+":"+oversample, flush = True)
            print('Loading Data')
            model = pickle.load(open('/home/hoyinchan/blue/Data/data2021/data2021/'+
                                     site_m+'/model_'+model_type+'_'+site_m+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))
            X_train3_d, X_test3_d, y_train_d, y_test_d = self.load_test_data(site_m, site_d, year, stg, fs, oversample, model_type)
            shapX = pd.concat([X_train3_d, X_test3_d])
            shapy = pd.concat([y_train_d, y_test_d])            

            if cat_feature_flag:
                cat_feature = [k for k in shapX.dtypes.keys() if k in self.select_features and shapX.dtypes[k]==bool]                
                shapX, shapy = self.gen_cat_sample(shapX, shapy, number_of_sample, cat_feature)
            else:
                shapX, shapy = self.gen_samples(shapX, shapy, number_of_sample, sample_x)
            
            if shapX.shape[0] == 0:
                print('No positive sample')
                return
                
#            shap = self.getshap2(X_train3_d, X_test3_d, model, n_jobs=n_jobs)
            print('Calculating SHAP')
            shap = self.getshap1(shapX, shapy, model, n_jobs=n_jobs)
            shapdf = pd.DataFrame(shap).iloc[:,:-1]
            shapdf.columns = shapX.columns
            shapdf.index = shapX.index
            
            if sample_y:
                filter_features = [x for x in shapX.columns if x in self.select_features]
                shapdf = shapdf[filter_features]
                shapX = shapX[filter_features]
        
            print('Transforming SHAP')        
            shap = self.shap_transform(shapX, shapdf, site_m, site_d, year, stg, fs, oversample, model_type, n_jobs=n_jobs)
            if save:           
                print('Saving SHAP')
                if cat_feature_flag:
                    shap.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+
                                   '/shapdataraw_catfeature_'+model_type+'_'+site_m+'_'+site_d+'_'+
                                   str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl')                    
                else:
                    shap.to_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+
                                   '/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+
                                   str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl')
            else:
                return shap
        except Exception as e:
            print(e)     

    def set_select_features(self, custom_features=None, top1=None, top2=None):
        if custom_features is None:
            self.select_features = list(self.get_top_top(top1,top2)['Feature Id'])
        else:
            self.select_features = custom_features
            
    def getshap1(self, shapX, shapy, model, n_jobs=1):
        def cal_shap(shapX, shapy):
            cat_features = model.get_cat_feature_indices()
            pshap = Pool(data=shapX, label=shapy, cat_features=cat_features) 
            return model.get_feature_importance(data=pshap, type='ShapValues',prettified=True)
        n=int(shapX.shape[0]/n_jobs)+1
        chunksX = [shapX[i:i+n].copy() for i in range(0,shapX.shape[0],n)]
        chunksy = [shapy[i:i+n].copy() for i in range(0,shapy.shape[0],n)]
        shap = Parallel(n_jobs=n_jobs)(delayed(cal_shap)(chunksX[i],chunksy[i]) for i in range(len(chunksX)))
        shap = np.concatenate(shap)
        return shap
    
    def getshap2(self, X_train3_d, X_test3_d, model, n_jobs=1):
        shapX = pd.concat([X_train3_d, X_test3_d])
        explainer = sp.Explainer(model, algorithm='tree')
        n=int(shapX.shape[0]/n_jobs)+1
        chunks = [shapX[i:i+n].copy() for i in range(0,shapX.shape[0],n)]
        shap = Parallel(n_jobs=n_jobs)(delayed(explainer.shap_values)(chunk) for chunk in chunks)
        shap = np.concatenate(shap)
        return shap

    def shap_transform(self, shapX, shap, site_m, site_d, year, stg, fs, oversample, model_type, n_jobs=1):
        def reformat(shapX, shap, i):
            tmpdf = pd.DataFrame(list(zip(list(range(shapX.shape[0])), shapX.iloc[:,i], shap.iloc[:,i])), columns =['ID', 'Name', 'val'])
            tmpdf['Feature'] = shapX.columns[i]
            return tmpdf
        shap_data_raw = Parallel(n_jobs=n_jobs)(delayed(reformat)(shapX, shap, i) for i in range(shapX.columns.shape[0]))
        shap_data_raw = pd.concat(shap_data_raw)    
        shap_data_raw['site_d'] = site_d
        shap_data_raw['site_m'] = site_m    
#        shap_data_raw['stg'] = site_m
#        shap_data_raw['fs'] = fs
#        shap_data_raw['oversample'] = oversample
#        shap_data_raw['model_type'] = model_type        
        # select = pd.DataFrame([[site_m,site_d,stg,fs,oversample,model_type]])
        # select.columns=['site_m','site_d','stg','fs','oversample','model_type']
        # roc = select.merge(self.barX, on=['site_m','site_d','stg','fs','oversample','model_type'], how='left').iloc[0,-1]
#        shap_data_raw['auc'] = roc
        return shap_data_raw

    def get_shap_all(self, n_jobs=1, save=False, rerun=False, sample_x=True, number_of_sample=50000, sample_y=True, top1=30, top2=20, single_site=False, sitein_m=None, sitein_d=None, cat_feature_flag=False):
        self.gen_feature_ranking_list()
        self.set_select_features(top1=top1, top2=top2)
        def run_if_not_exists(site_m, site_d, stg, fs, oversample, model_type, n_jobs, save, rerun, sample_x, number_of_sample, sample_y, cat_feature_flag):
            if cat_feature_flag:
                filepath = '/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/shapdataraw_catfeature_'+model_type+'_'+site_m+'_'+site_d+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'       
            else:
                filepath = '/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
            if rerun or not exists(filepath):
                return self.getshap(site_m, site_d, 3000, stg, fs, oversample, model_type, n_jobs=n_jobs, save=save, sample_x=sample_x, number_of_sample=number_of_sample, sample_y=sample_y, cat_feature_flag=cat_feature_flag) 
            
        if single_site:
            print('Running getshap single site '+'on site '+sitein_m+'/'+sitein_d, flush = True)
            [run_if_not_exists(sitein_m, sitein_d, stg, fs, oversample, model_type, n_jobs, True, rerun, sample_x, number_of_sample, sample_y, cat_feature_flag)
                        for stg in self.stgs for fs in self.fss for oversample in self.oversamples for model_type in self.model_types]            
            return
            
        if save:
            [run_if_not_exists(site_m, site_d, stg, fs, oversample, model_type, n_jobs, save, rerun, sample_x, number_of_sample, sample_y, cat_feature_flag)
                        for site_m in self.sites for site_d in self.sites for stg in self.stgs 
                        for fs in self.fss for oversample in self.oversamples for model_type in self.model_types]
        else:
            shap = [run_if_not_exists(site_m, site_d, stg, fs, oversample, model_type, n_jobs, save, rerun, sample_x, number_of_sample, sample_y, cat_feature_flag)
                            for site_m in self.sites for site_d in self.sites for stg in self.stgs 
                            for fs in self.fss for oversample in self.oversamples for model_type in self.model_types]
            self.shap = pd.concat(shap)            
            
    def gen_feature_ranking_list(self):
        feature_ranking_list = list()
        for site_m in self.sites:
            for stg in self.stgs:
                for fs in self.fss:
                    for oversample in self.oversamples:
                        for model_type in self.model_types:
                            model = pickle.load(open('/home/hoyinchan/blue/Data/data2021/data2021/'+
                                                                 site_m+'/model_'+model_type+'_'+site_m+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))            
                            feature_ranking = model.get_feature_importance(prettified=True).reset_index()
                            feature_ranking = feature_ranking[feature_ranking['Importances']>0].drop('Importances',axis=1)
                            feature_ranking.columns = ['rank', 'Feature Id']
                            feature_ranking['site'] = site_m
                            feature_ranking['stg'] = stg
                            feature_ranking['fs'] = fs
                            feature_ranking['oversample'] = oversample
                            feature_ranking['model_type'] = model_type
                            feature_ranking_list.append(feature_ranking)                        
        self.feature_ranking=pd.concat(feature_ranking_list)       
        
    def get_top_top(self, top_1, top_2):
        xxx = self.feature_ranking[np.logical_and(self.feature_ranking['fs']=='nofs',self.feature_ranking['stg']=='stg01')]
        return xxx[xxx['rank']<=top_1][['Feature Id','rank']].groupby('Feature Id').count().reset_index().sort_values('rank',ascending=False).head(top_2)
    
    def generate_interaction(self, site_m, stg, fs, oversample, model_type):
        try:
            model = pickle.load(open('/home/hoyinchan/blue/Data/data2021/data2021/'+
                                         site_m+'/model_'+model_type+'_'+site_m+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))

            interact_rank = model.get_feature_importance(type='Interaction', prettified=True)

            feature_map = pd.DataFrame(model.feature_names_).reset_index()

            interact_rank1 = interact_rank.merge(feature_map, left_on='First Feature Index', right_on = 'index', how='left').drop('index',axis=1)
            interact_rank2 = interact_rank1.merge(feature_map, left_on='Second Feature Index', right_on = 'index', how='left').drop('index',axis=1)
            interact_rank2 = interact_rank2.drop(['First Feature Index', 'Second Feature Index'],axis=1)
            interact_rank2.columns = ['Interaction', 'First Feature', 'Second Feature']
            interact_rank2 = interact_rank2[['First Feature', 'Second Feature', 'Interaction']]
            interact_rank3 = interact_rank2.copy()

            # loincmap3 =pd.read_csv(self.home_directory+'loinc/AccessoryFiles/GroupFile/GroupLoincTerms.csv') 
            # loincmap3l = loincmap3[['LoincNumber','LongCommonName']]
            # loincmap3g = loincmap3[['GroupId','LongCommonName']]
            # loincmap3g.columns = ['LoincNumber','LongCommonName']
            # loincmap3s = pd.concat([loincmap3l, loincmap3g])

            # interact_rank3['First Feature Loinc'] = [x.split('::')[1].split('(')[0] if 'LAB' in x else x for x in interact_rank3['First Feature']]
            # interact_rank3x = interact_rank3.merge(loincmap3s.drop_duplicates(), left_on='First Feature Loinc', right_on='LoincNumber', how='left').drop(['First Feature Loinc', 'LoincNumber'],axis=1).copy()
            # interact_rank3x['Second Feature Loinc'] = [x.split('::')[1].split('(')[0] if 'LAB' in x else x for x in interact_rank3x['Second Feature']]
            # interact_rank3x = interact_rank3x.merge(loincmap3s.drop_duplicates(), left_on='Second Feature Loinc', right_on='LoincNumber', how='left').drop(['Second Feature Loinc', 'LoincNumber'],axis=1).copy()

            interact_rank3x=interact_rank3
            
            interact_rank3x['site'] = site_m
            interact_rank3x['stg'] = stg
            interact_rank3x['fs'] = fs
            interact_rank3x['rank'] = interact_rank3x['Interaction'].rank(method='min', ascending=False)             
            return interact_rank3x
        except:
            pass
        
    def generate_interaction_all(self, n_jobs=1):
            interaction = Parallel(n_jobs=n_jobs)(delayed(self.generate_interaction)(site_m, stg, fs, oversample, model_type)
                            for site_m in self.sites for stg in self.stgs 
                            for fs in self.fss for oversample in self.oversamples for model_type in self.model_types)
            self.interaction = pd.concat(interaction)        

    def generate_feature_ranking_table(self, top=10):
        feature_table = shap3.feature_ranking.copy()
        feature_table['Feature Id'] = feature_table['Feature Id'].str.split('(').str[0]
        feature_table = feature_table.sort_values('rank').groupby('site').head(10).pivot(index='site', columns='rank', values='Feature Id')        
        feature_table.columns = feature_table.columns+1
        return feature_table
            
    def get_rank_shap(self, site_m, stg, fs, oversample, model_type):
        site_d = site_m
        filepath = '/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
        shap = pd.read_pickle(filepath)
        shap = shap[['Feature', 'val']]
        shap['val'] = abs(shap['val'])
        shaprank = shap.groupby('Feature').mean().reset_index().sort_values('val')
        shaprank['site'] = site_m
        shaprank['stg'] = stg
        shaprank['fs'] = fs
        shaprank['rank'] = shaprank['val'].rank(method='min', ascending=False) 
        return shaprank

    def get_rank_importance(self, site_m, stg, fs, oversample, model_type):
        try:
            model = pickle.load(open('/home/hoyinchan/blue/Data/data2021/data2021/'+
                                 site_m+'/model_'+model_type+'_'+site_m+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))
            shaprank = model.get_feature_importance(prettified=True)
            shaprank['Feature'] = shaprank['Feature Id']
            shaprank = shaprank[['Feature', 'Importances']]
            shaprank['site'] = site_m
            shaprank['rank'] = shaprank['Importances'].rank(method='min', ascending=False)
            return shaprank
        except:
            pass
        
    def get_feature_rank_all(self):
        XX = Parallel(n_jobs=30)(delayed(self.get_rank_importance)(site_m, stg, fs, oversample, model_type) 
                                             for site_m in self.sites for stg in self.stgs 
                                             for fs in self.fss for oversample in self.oversamples for model_type in self.model_types)  
        self.feature_importance=pd.concat(XX)

    def define_barX(self):
        self.barX = self.roc_to_table()

        
    def collect_shap(self):
        from os.path import exists
        shap_list = list()
        for site_m in self.sites:
            for site_d in self.sites:
                for stg in self.stgs:
                    for fs in self.fss:
                        for oversample in self.oversamples:
                            for model_type in self.model_types:
                                filepath = '/home/hoyinchan/blue/Data/data2021/data2021/'+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(3000)+'_'+stg+'_'+fs+'_'+oversample+'.pkl'
                                if exists(filepath):
                                    shap = pd.read_pickle(filepath)
                                    shap_list.append(shap)
                                else:
                                    print('Missing ' + site_m+'/'+ site_d+'/'+ stg+'/'+ fs)
                
        shap_list = pd.concat(shap_list)
        
        ####
        self.define_barX()
        roc_data = self.barX[np.logical_and(self.barX['stg']=='stg01', self.barX['fs']=='nofs')]
#        roc_data = self.barX
        shap_list = shap_list.merge(roc_data[['site_d', 'site_m', 'roc']], on=['site_d', 'site_m'], how='left')
        shap_list = shap_list.dropna()

        g = shap_list[['Feature', 'site_d', 'site_m', 'Name', 'val']].groupby(['Feature', 'site_d', 'site_m']).quantile([0.05, 0.95])
        gg = g.reset_index()
        gg['level_3'] = gg['level_3'].astype(str)
        gg = gg.pivot(index=['Feature', 'site_d', 'site_m'], values=['Name', 'val'], columns='level_3')
        gg.columns = gg.columns.map('_'.join).str.strip('_')
        gg = gg.reset_index()
        shap_list = shap_list.merge(gg, on = ['Feature', 'site_d', 'site_m'], how='left')
        
        cnt = shap_list[['Feature', 'Name']].drop_duplicates().groupby(['Feature']).count()
        cat_feature = cnt[cnt['Name']==2].index
        cat_feature = '|'.join(cat_feature)
        tmp_shap_list = shap_list[np.logical_and(shap_list['Feature'].str.contains(cat_feature), shap_list['Name']==1)].copy()
        
        shap_list['drop'] = np.logical_or(shap_list['Name']<shap_list['Name_0.05'], shap_list['Name']>shap_list['Name_0.95'])
        shap_list['drop_age'] = np.logical_or(shap_list['Name']<18, shap_list['Name']>89)
        shap_list.loc[shap_list['Feature']=='AGE', 'drop'] = shap_list.loc[shap_list['Feature']=='AGE', 'drop_age']
        shap_list = shap_list[shap_list['drop']==False]
        shap_list = shap_list.drop(['Name_0.05','Name_0.95','val_0.05','val_0.95','drop','drop_age'], axis=1)

        shap_list = pd.concat([shap_list, tmp_shap_list]) 
        shap_list['roc2'] = 1-shap_list['roc']
        shap_list = shap_list.drop(['Name_0.05','Name_0.95','val_0.05','val_0.95','drop','drop_age'], axis=1, errors='ignore')
        shap_list = shap_list.drop_duplicates()
        
        shap_list.to_parquet('/home/hoyinchan/blue/Data/data2021/data2021/shapalltmp.parquet')        
        
    def save(self):
        with open(self.datafolder+'shap_'+self.stgs[0]+'_'+self.fss[0]+'.pkl', 'wb') as f:
            pickle.dump(self, f)             
        
    def get_shap_all2(self, n_jobs=1, rerun=False, sample_x=True, number_of_sample=50000, sample_y=True, top1=30, top2=30, cat_feature_flag=False):
        sitesp = list(itertools.product(self.sites,self.sites))
        sitesp.reverse()        
        Parallel(n_jobs=n_jobs)(delayed(self.get_shap_all)(n_jobs=1, save=True, rerun=rerun, sample_x=sample_x, number_of_sample=number_of_sample, sample_y=sample_y, top1=top1, top2=top2, single_site=True, sitein_m=site[0], sitein_d=site[1], cat_feature_flag=cat_feature_flag) for site in sitesp)

    def average_transportability(self):
        self.barX['roc'] = self.barX['roc'].astype(float)
        return pd.DataFrame(np.round(self.barX[self.barX['site_m']!=self.barX['site_d']].groupby('site_d')['roc'].mean(),2))        
    
    def average_adaptation(self):
        self.barX['roc'] = self.barX['roc'].astype(float)
        return pd.DataFrame(np.round(self.barX[self.barX['site_m']!=self.barX['site_d']].groupby('site_m')['roc'].mean(),2))    
## Batch process    
    
if __name__ == "__main__":
    datafolder = '/home/hoyinchan/blue/Data/data2021/data2021/'
    home_directory = "/home/hoyinchan/code/AKI_CDM_PY/"
    shap1 = generateSHAP()
    with open(datafolder+"shap.pkl", "rb") as f:
        shap2 = pickle.load(f)
#    shap1.copy(shap2)
    shap1.get_shap_all2(n_jobs=30, rerun=True, sample_x=True, number_of_sample=50000, sample_y=True, top1=30, top2=23)
#    shap1.cross_roc_all(n_jobs=30)
#    shap1.save_cross_roc()

