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
#from statsmodels.tsa.vector_ar.var_model import VARY
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
import scipy
import datetime

from joblib import Parallel, delayed

import scipy
import forestplot as fp

import requests
import re
import statsmodels.api as sm
from PIL import Image

import pingouin as pg
import statsmodels

class mycorr2:
    
    def __init__(self):
        self.site = None
        self.lab = None
        self.labX = None
        self.vital = None
        self.vitalX = None
        self.onset = None
        self.medX = None
        self.loinc1 = None
        self.loinc2 = None
        self.loinc3 = None
        self.medcode_rx = dict()
        self.medcode_nd = dict()

    def copy(self, new_instance):
        # self.site = new_instance.site
        # self.labX = new_instance.labX.copy() if new_instance.labX is not None else None
        # self.vital = new_instance.vital.copy() if new_instance.vital is not None else None
        # self.vitalX = new_instance.vitalX.copy() if new_instance.vitalX is not None else None
        # self.onset = new_instance.onset.copy() if new_instance.onset is not None else None
        # self.medX = new_instance.medX.copy() if new_instance.medX is not None else None
        
        # self.loinc1 = new_instance.loinc1
        # self.loinc2 = new_instance.loinc2
        # self.loinc3 = new_instance.loinc3
        
        self.one_table = new_instance.one_table
    
    def getdata(self, site='KUMC'):
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
        
        onset = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/p0_onset_'+site+'.pkl')
        years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
        bt_list = list()

        for year in years:
            try:
                data = pd.read_pickle('/home/hoyinchan/blue/Data/data2021/data2021/'+site+'/bt3_'+site+'_'+str(year)+'.pkl')
                data = flag_convert(data, 'stg01')
                bt_list.append(data.copy())
            except:
                print(str(year)+' not exists')
                
        target_col = list()
        for bt in bt_list:
            for y in self.loinc1 + self.loinc2 + self.loinc3 + ['SYSTOLIC', 'DIASTOLIC', 'FLAG']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
                target_col = target_col + [x for x in bt.columns if y in x]
        target_col = np.unique(target_col)

        bt2 = list()
        for bt in bt_list:
            bt2tmp = bt.loc[:, bt.columns.isin(target_col)].copy()
            bt2tmp = bt2tmp[bt2tmp.columns[bt2tmp.dtypes != bool]]            
            bt2.append(bt2tmp)

        bt2 = pd.concat(bt2)

        newidx = dict()
        for y in self.loinc1 + self.loinc2 + self.loinc3 + ['SYSTOLIC', 'DIASTOLIC', 'FLAG']:
            try:
                tc2 = [x for x in bt2.columns if y in x]
                newidx[y] = bt2[tc2].notnull().sum().idxmax()
            except:
                pass

        for y in list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
            try:
                newidx[y] = [x for x in bt.columns if (('MED' in x) & (y in x))][0]
            except:
                pass    

        bt3 = bt[newidx.values()]

        bt3 = bt3.rename(dict((v, k) for k, v in newidx.items()),axis=1)
        bt3 = bt3.replace({True:1, False:0})

        self.one_table_original = bt3.copy()

    def set_datarangemode(self, datarange='full'):

        if datarange == 'upper':
            self.one_table = self.one_table_original[(self.one_table_original[self.loinc2[0]]>=self.threshold_upper[self.loinc2[0]])].copy()
        elif datarange == 'lower':
            self.one_table = self.one_table_original[(self.one_table_original[self.loinc2[0]]<=self.threshold_lower[self.loinc2[0]])].copy()
        else:    
            self.one_table = self.one_table_original.copy()
            


            
    def calculate_corr3(self):
        resdf = list()
        self.one_table_w = self.one_table.drop('FLAG',axis=1).notnull().merge(self.one_table[['FLAG']], left_index=True, right_index=True)
        self.one_table_w['FLAG_COUNT'] = 1
        self.one_table_w2 = self.one_table_w.groupby('FLAG').sum()
        self.one_table_w2 = self.one_table_w2.div(self.one_table_w2['FLAG_COUNT'], axis=0).drop('FLAG_COUNT',axis=1)
        self.one_table2 = self.one_table
        self.one_table_w2.columns= self.one_table_w2.columns+'_w'
        self.one_table_w2
        self.one_table2 = self.one_table2.merge(self.one_table_w2.reset_index(),how='left')
                
        for x in self.loinc1+['SYSTOLIC', 'DIASTOLIC']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
            for y in self.loinc2+self.loinc3:
                try:
                    xloinc = x
                    yloinc = y

                    tmp_table = self.one_table2[[xloinc, yloinc, xloinc+'_w']]
                    tmp_table = tmp_table.dropna().reset_index(drop=True)
                    mywstat = statsmodels.stats.weightstats.DescrStatsW(tmp_table[[xloinc, yloinc]], weights=list(tmp_table[xloinc+'_w']))
                    #scipy.stats.pearsonr(tmp_table[xloinc], tmp_table[yloinc])
                    resv = pd.DataFrame([[xloinc, yloinc, mywstat.corrcoef[0,1], 0, min(mywstat.corrcoef[0,1]*np.array([.9999,1.0001])), max(mywstat.corrcoef[0,1]*np.array([.9999,1.0001]))]], columns=['feature1', 'feature2', 'rho', 'pvalue', '95ci_low', '95ci_high'])
                    resdf.append(resv)
                except:
                    pass
        try:
            xloinc = self.loinc2[0]
            yloinc = self.loinc3[0]
            tmp_table = self.one_table2[[xloinc, yloinc, xloinc+'_w']]
            tmp_table = tmp_table.dropna().reset_index(drop=True)
            mywstat = statsmodels.stats.weightstats.DescrStatsW(tmp_table[[xloinc, yloinc]], weights=list(tmp_table[xloinc+'_w']))
            #scipy.stats.pearsonr(tmp_table[xloinc], tmp_table[yloinc])
            resv = pd.DataFrame([[xloinc, yloinc, mywstat.corrcoef[0,1], 0, min(mywstat.corrcoef[0,1]*np.array([.9999,1.0001])), max(mywstat.corrcoef[0,1]*np.array([.9999,1.0001]))]], columns=['feature1', 'feature2', 'rho', 'pvalue', '95ci_low', '95ci_high'])
            resdf.append(resv)
        except:
            pass
        self.resdf3 = pd.concat(resdf)      

        
    def calculate_corr(self):
        resdf = list()
        for x in self.loinc1+['SYSTOLIC', 'DIASTOLIC']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
            for y in self.loinc2+self.loinc3:
                try:
                    xloinc = x
                    yloinc = y
                    xtmp = self.one_table[[xloinc,yloinc]].dropna()
                    res = scipy.stats.pearsonr(xtmp[xloinc], xtmp[yloinc])
                    resv = pd.DataFrame([[xloinc, yloinc, res.statistic, res.pvalue, res.confidence_interval(confidence_level=0.95).low, res.confidence_interval(confidence_level=0.95).high]], columns=['feature1', 'feature2', 'rho', 'pvalue', '95ci_low', '95ci_high'])
                    resdf.append(resv)
                except:
                    pass
        try:
            xloinc = self.loinc2[0]
            yloinc = self.loinc3[0]
            xtmp = self.one_table[[xloinc,yloinc]].dropna()
            res = scipy.stats.pearsonr(xtmp[xloinc], xtmp[yloinc])
            resv = pd.DataFrame([[xloinc, yloinc, res.statistic, res.pvalue, res.confidence_interval(confidence_level=0.95).low, res.confidence_interval(confidence_level=0.95).high]], columns=['feature1', 'feature2', 'rho', 'pvalue', '95ci_low', '95ci_high'])
            resdf.append(resv)
        except:
            pass
        self.resdf = pd.concat(resdf)          
        
    def calculate_corr2(self):
        df_t = list()
        x = self.loinc2[0]
        y = self.loinc3[0]
        for z in self.loinc1+['SYSTOLIC', 'DIASTOLIC']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
            try:
                tmpone_table = self.one_table[[x,y,z]].copy().dropna()
                X = tmpone_table[[x]]
                X2 = sm.add_constant(X)
                if y == 'FLAG':
                    rxy = sm.Logit(tmpone_table[y], X2).fit(disp=0)                    
                else:
                    rxy = sm.OLS(tmpone_table[y], X2).fit()
                
                Xz = tmpone_table[[x, z]]
                Xz2 = sm.add_constant(Xz)

                if y == 'FLAG':
                    rxyz = sm.Logit(tmpone_table[y], Xz2).fit(disp=0)
                else:
                    rxyz = sm.OLS(tmpone_table[y], Xz2).fit()

                txy = rxy.summary2().tables[1].loc[[x],:]
                txyz = rxyz.summary2().tables[1].loc[[x],:]
                txy.columns = txy.columns+'_xy'
                txyz.columns = txyz.columns+'_xyz'

                t = pd.concat([txy, txyz],axis=1)
                t['ratio'] = (t['Coef._xyz'] - t['Coef._xy'])/t['Coef._xyz']
                t.index = [z]
                df_t.append(t)
            except:
                pass
        #pd.concat(df_t)[['ratio', 'Coef._xy', 'P>|t|_xy', 'Coef._xyz', 'P>|t|_xyz']]
        df_t = pd.concat(df_t)
        self.res_df2 = df_t

    def set_loincs_pair(self, loinc1 = None, loinc2 = None, loinc3 = None, vitalcode = None, medcode_rx = None, medcode_nd = None):
        self.loinc1 = ['2157-6', #creatine kinase
                  '1920-8', #AST      
                  '2532-0', #LDH
                  '4542-7', #Haptoglobin 
                  '3084-1'] #uric acid

        self.loinc2_list = ['2823-3', '17861-6', '2951-2']
        
#        self.loinc2 = ['2823-3'] # potassium
#        self.loinc2 = ['17861-6'] # calcium
#        self.loinc2 = ['2951-2'] # sodium       
        
        self.loinc3_list = ['2160-0', 'FLAG'] #sCr
        self.vitalcode = vitalcode
        if medcode_rx is None:
            self.medcode_rx = dict()
        else:
            self.medcode_rx = medcode_rx
        if medcode_nd is None:
            self.medcode_nd = dict()
        else:
            self.medcode_nd = medcode_nd
            
        # medcode_nd = {'338011704': 'A', '338114803': 'B'}
        # medcode_rx = {'338011704': 'A', '338114803': 'B'}
        
    def set_range(self):
        
        self.threshold_lower = dict()
        self.threshold_upper = dict()

        self.threshold_lower['2823-3'] = 3.2
        self.threshold_upper['2823-3'] = 4.4
        
        self.threshold_lower['2951-2'] = 132
        self.threshold_upper['2951-2'] = 144
        
        self.threshold_lower['17861-6'] = 8
        self.threshold_upper['17861-6'] = 9.25                     

    def pcorr_custom(self):
        feature1 = self.loinc2
        feature2 = self.loinc3

        def pcorr_custom_sub(df, feature1, feature2, covar):
            pcorrdf = pg.partial_corr(data=df, x=feature1, y=feature2, covar=covar).round(4)
            pcorrdf['feature1']=feature1
            pcorrdf['feature2']=feature2
            pcorrdf['covar']=covar
            return pcorrdf

        corrdf = pg.pairwise_corr(data=self.one_table, columns = [feature1, feature2]).round(4)
        pcorrdf = pd.concat([pcorr_custom_sub(self.one_table, feature1, feature2, covar) for covar in [x for x in self.loinc1+['SYSTOLIC', 'DIASTOLIC']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))) if x in self.one_table.columns]])  
        pcorrdf['n0'] = corrdf.loc[0,'n']
        pcorrdf['r0'] = corrdf.loc[0,'r']
        pcorrdf['CI95%0down'] = corrdf.loc[0,'CI95%'][0]
        pcorrdf['CI95%0up'] = corrdf.loc[0,'CI95%'][1]
        pcorrdf['CI95%down'], pcorrdf['CI95%up'] = zip(*list(pcorrdf['CI95%'].values))
        pcorrdf['significant'] = (pcorrdf['r0'] <= pcorrdf['CI95%down']) | (pcorrdf['r0'] >= pcorrdf['CI95%up'])
        pcorrdf['CI95%down_adj'] = pcorrdf['CI95%down']-pcorrdf['r0']
        pcorrdf['CI95%up_adj'] = pcorrdf['CI95%up']-pcorrdf['r0']
        pcorrdf['pdiff'] = pcorrdf['r']-pcorrdf['r0']
        
        
        pcorrdf['rho'] = pcorrdf['pdiff']
        pcorrdf['95ci_low'] = pcorrdf['CI95%down_adj']
        pcorrdf['95ci_high'] = pcorrdf['CI95%up_adj']
        pcorrdf['rho'] = pcorrdf['pdiff']
        pcorrdf['pvalue'] = (~pcorrdf['significant']).astype(int)
        
        self.res_df3 =  pcorrdf        

    def calculate_site(self, site):
        # import mycorr2
        # importlib.reload(mycorr2)
        myco = mycorr2()
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        with open(datafolder+'myco2_tmp_'+site+'.pkl', "rb") as f:
            myco2 = pickle.load(f)
        myco.copy(myco2)

        myco.set_loincs_pair()
        for loinc2 in myco.loinc2_list:
            for loinc3 in myco.loinc3_list:
                myco.loinc2 = [loinc2]
                myco.loinc3 = [loinc3]        
        #        drugs = pd.read_csv('druglist.csv').drop('Unnamed: 0',axis=1)
                drugs = pd.read_csv('atclist.txt')
                drugs = drugs[drugs['target'] == myco.loinc2[0]]
                rxdrugs = drugs[drugs['type']=='rx'][['code', 'name']]
                medcode_rx = rxdrugs.set_index('code')['name'].to_dict()
                ndcdrugs = drugs[drugs['type']=='ndc'][['code', 'name']]
                medcode_nd = ndcdrugs.set_index('code')['name'].to_dict()        
                myco.set_loincs_pair(loinc1=myco.loinc1, loinc2=myco.loinc2, loinc3=myco.loinc3, medcode_rx=medcode_rx, medcode_nd=medcode_nd)

                myco.getdata(site=site)
                myco.set_range()

                
                datarange = 'full'
                myco.set_datarangemode(datarange=datarange)
                myco.calculate_corr()
                myco.resdf['site'] = site        
                myco.resdf['target'] = myco.loinc2[0]                
                myco.resdf.to_csv('myco2_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')   

                myco.calculate_corr3()
                myco.resdf3['site'] = site        
                myco.resdf3['target'] = myco.loinc2[0]                
                myco.resdf3.to_csv('myco2_corr4_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')          
                
                try:
                    myco.calculate_corr2()
                    myco.res_df2['site'] = site        
                    myco.res_df2['target'] = myco.loinc2[0]                
                    myco.res_df2.to_csv('myco2_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')   
                    
                    myco.pcorr_custom()              
                    myco.res_df3['site'] = site        
                    myco.res_df3['target'] = myco.loinc2[0]                
                    myco.res_df3.to_csv('myco2_corr3_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')   
                    
                except:
                    pass

                datarange = 'upper'
                myco.set_datarangemode(datarange=datarange)
                myco.calculate_corr()
                myco.resdf['site'] = site
                myco.resdf['target'] = myco.loinc2[0]                        
                myco.resdf.to_csv('myco2_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')        
                
                myco.set_datarangemode(datarange=datarange)
                myco.calculate_corr3()
                myco.resdf3['site'] = site        
                myco.resdf3['target'] = myco.loinc2[0]                
                myco.resdf3.to_csv('myco2_corr4_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')          
                
                try:        
                    myco.calculate_corr2()
                    myco.res_df2['site'] = site        
                    myco.res_df2['target'] = myco.loinc2[0]                
                    myco.res_df2.to_csv('myco2_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')      
                    
                    myco.pcorr_custom()              
                    myco.res_df3['site'] = site        
                    myco.res_df3['target'] = myco.loinc2[0]                
                    myco.res_df3.to_csv('myco2_corr3_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')                       
                    
                except:
                    pass

                datarange = 'lower'
                myco.set_datarangemode(datarange=datarange)        
                myco.calculate_corr()
                myco.resdf['site'] = site
                myco.resdf['target'] = myco.loinc2[0]                        
                myco.resdf.to_csv('myco2_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')      
                
                myco.set_datarangemode(datarange=datarange)
                myco.calculate_corr3()
                myco.resdf3['site'] = site        
                myco.resdf3['target'] = myco.loinc2[0]                
                myco.resdf3.to_csv('myco2_corr4_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')                          
                
                try:
                    myco.calculate_corr2()
                    myco.res_df2['site'] = site        
                    myco.res_df2['target'] = myco.loinc2[0]                
                    myco.res_df2.to_csv('myco2_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')   
                    
                    myco.pcorr_custom()              
                    myco.res_df3['site'] = site        
                    myco.res_df3['target'] = myco.loinc2[0]                
                    myco.res_df3.to_csv('myco2_corr3_'+datarange+'_'+site+'_'+myco.loinc2[0]+'_'+myco.loinc3[0]+'.csv')                       
                    
                except:
                    pass
        return myco            
    
    def calculate_all(self):
        sites = ['MCW', 'UMHC', 'UNMC', 'UofU', 'UTHSCSA', 'KUMC', 'UTSW', 'UIOWA', 'UPITT', 'IUR']
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        for site in sites:
            try:
                self.calculate_site(site)
            except:
                pass
            
    def generate_drug_csv(self):
        drugdfs = list()

        # Potassium
        druglist = ['Diphenoxylate', 'loperamide', 'Furosemide', 'bumetanide', 'chlorothiazide', 'metolazone', 'sacubitril-valsartan', 'spironolcatone', 'epleronone']
        target = '2823-3'
        df_drug = myco.getcodefromlist(druglist)
        df_drug['target'] = target
        drugdfs.append(df_drug.copy())

        #Sodium
        druglist = ['tolvaptan', 'Furosemide', 'bumetanide', 'torsemide', 'hydrochlorothiazide', 'chlorothiazide', 'chlorthalidone', 'metolazone']
        target = '2951-2'
        df_drug = myco.getcodefromlist(druglist)
        df_drug['target'] = target
        drugdfs.append(df_drug.copy())

        #Calcium
        druglist = ['Pamidronate', 'etidronate', 'zoledronate']
        target = '17861-6'
        df_drug = myco.getcodefromlist(druglist)
        df_drug['target'] = target
        drugdfs.append(df_drug.copy())

        pd.concat(drugdfs).to_csv('druglist.csv')
        
    def getcodefromlist(self, druglist):
        drugdfs = list()
        for drug in druglist:
            rxcui, ndc = self.getcode(drug)
            df_rxcui = pd.DataFrame(rxcui)
            df_rxcui.columns = ['code']
            df_rxcui['type'] = 'rx'
            df_rxcui['name'] = drug
            
            df_ndc = pd.DataFrame(ndc)
            df_ndc.columns = ['code']
            df_ndc['type'] = 'ndc'
            df_ndc['name'] = drug
            
            drugdfs.append(pd.concat([df_rxcui, df_ndc]).copy())
        return pd.concat(drugdfs)
    
    def getcode(self, drugname):
        response = requests.get("https://rxnav.nlm.nih.gov/REST/drugs.json?name="+drugname)
        rxcui = re.findall("\"rxcui\":\"(.*?)\"", response.text)
        ndc = list()
        for rx in rxcui:
            try:
                response2 = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui/"+rx+"/ndcs.json")
                ndc = ndc + response2.json()['ndcGroup']['ndcList']['ndc']
            except:
                pass
        ndc = np.unique(ndc)
        return np.array(rxcui), ndc
    
    def corr2_all2(self, prefix='myco_corr2', outfile='mlr6_cofounder2.csv'):
        files = [x for x in os.listdir() if prefix in x ]

        files_u = [x for x in files if 'upper' in x ]
        files_l = [x for x in files if 'lower' in x ]
        files_f = [x for x in files if 'full' in x ]

        df_u = list()
        for file in files_u:
            tmp = pd.read_csv(file)
            df_u.append(tmp)
        df_u = pd.concat(df_u)
        df_u['range'] = 'upper'

        df_l = list()
        for file in files_l:
            tmp = pd.read_csv(file)
            df_l.append(tmp)
        df_l = pd.concat(df_l)
        df_l['range'] = 'lower'

        df_f = list()
        for file in files_f:
            tmp = pd.read_csv(file)
            df_f.append(tmp)
        df_f = pd.concat(df_f)
        df_f['range'] = 'full'

        df = pd.concat([df_u, df_l, df_f])
        df = df[['ratio', 'Unnamed: 0', 'Coef._xy', 'P>|t|_xy', 'target', 'range', 'site']]

        np.round(df[(abs(df['ratio'])>=0.1) & (df['P>|t|_xy']<=0.1)],6).to_csv(outfile)

    def calculate_all2_all(self):
        self.set_loincs_pair()
        for loinc2 in self.loinc2_list:
            for loinc3 in self.loinc3_list:
                self.calculate_all2(loinc2=loinc2, loinc3=loinc3)
        
    def calculate_all2(self, loinc2='2823-3', loinc3='FLAG', prefix='myco2_corr_', plot_prefix="plot2_"):
        def cal_temp_stat(datarange):
            sites = ['MCW', 'UMHC', 'UNMC', 'UofU', 'UTHSCSA', 'KUMC', 'UTSW', 'UIOWA', 'UPITT', 'IUR']
            tmps = list()
            for site in sites:
                try:
                    tmp = pd.read_csv(prefix+datarange+'_'+site+'_'+loinc2+'_'+loinc3+'.csv')
                    tmps.append(tmp)
                except:
                    pass
            tmps = pd.concat(tmps)
            tmps = tmps.drop('Unnamed: 0',axis=1)
            print(tmps.columns)
            
            def pmean(df):
                return scipy.stats.ttest_1samp(df, 0)[1]
            tmpsmeanstat = tmps[['feature1','feature2','rho']].groupby(['feature1', 'feature2']).agg([np.nanmean, np.nanstd, pmean]).reset_index()
            tmpsmeanstat.columns = ['feature1','feature2','rho','std', 'pvalue']
            tmpsmeanstat['95ci_low'] = tmpsmeanstat['rho']-2*tmpsmeanstat['std']
            tmpsmeanstat['95ci_high'] = tmpsmeanstat['rho']+2*tmpsmeanstat['std']
            tmpsmeanstat['site'] = 'MEAN'
            tmpsmeanstat = tmpsmeanstat.drop('std',axis=1)
        
            tmpsmeanstatzero = tmpsmeanstat.copy()
            tmpsmeanstatzero['rho'] = 0
            tmpsmeanstatzero['pvalue'] = 0
            tmpsmeanstatzero['95ci_low'] = 0
            tmpsmeanstatzero['95ci_high'] = 0
            tmpsmeanstatzero['site'] = 'ZERO'
            tmps2 = pd.concat([tmps, tmpsmeanstat, tmpsmeanstatzero])
            tmps2['95ci_min'] = tmps2[['95ci_low','95ci_high']].min(axis=1)
            tmps2['95ci_max'] = tmps2[['95ci_low','95ci_high']].max(axis=1)
            tmps3  = tmps2[['feature1','feature2']].drop_duplicates()
            return tmps2, tmps3
        
        tmps2f, tmps3f = cal_temp_stat(datarange='full')
        tmps2u, tmps3u = cal_temp_stat(datarange='upper')
        tmps2l, tmps3l = cal_temp_stat(datarange='lower')

        tmps2u = tmps2u[tmps2u['site']!='ZERO']
        tmps2f = tmps2f[tmps2f['site']!='ZERO']
        tmps3 = pd.concat([tmps3f, tmps3u, tmps3l]).drop_duplicates()
        tmps2f['range'] = 'full'
        tmps2u['range'] = 'upper'
        tmps2l['range'] = 'lower'
        tmps2 = pd.concat([tmps2f, tmps2u, tmps2l]).drop_duplicates()        

        
        tmps2['ifsig'] = tmps2['pvalue']<=0.05
        count1 = tmps2[['feature1', 'feature2', 'site', 'range', 'ifsig']].groupby(['feature1', 'feature2', 'range', 'ifsig']).count().reset_index()
        countt = tmps2[['feature1', 'feature2', 'site', 'range']].groupby(['feature1', 'feature2', 'range']).count().reset_index()
        count1.columns = ['feature1', 'feature2', 'range', 'ifsig', 'count1']
        countt.columns = ['feature1', 'feature2', 'range', 'countt']
        countf = count1.merge(countt, how='left', on = ['feature1', 'feature2', 'range'])
        countf = countf[countf['ifsig']]
        dict_lab = {'2157-6':'creatineKinase(HK)',
                  '1920-8':'AST(HK)',      
                  '2532-0':'LDH(HK)',
                  '4542-7':'Haptoglobin(HK)', 
                  '3084-1':'uricAcid(HK)',
                '2823-3':'potassium',
                '17861-6':'calcium',
                '2951-2':'sodium', 
                '2160-0':'sCr',
                'FLAG':'FLAG'}
        
        dict_med = {'A07DA': 'DiphenoxylateLoperamide(LK)',
                      'C03CA': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03CB': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03EB': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03AA': 'hydrochlorothiazide(LNa)',
                      'C03AB': 'hydrochlorothiazide(LNa)',
                      'C03AH': 'chlorothiazide(LNa)',
                      'C03BA': 'chlorthalidone(LNa)',
                      'C03EA': 'hydrochlorothiazide(LNa)',
                      'C09DX': 'SacubitrilValsartan(HK)',
                      'C03DA': 'eplerenone(HK)',
                      'C03XA': 'tolvaptan(LNa)',
                      'C09XA': 'hydrochlorothiazide(LNa)',
                      'C09DX': 'hydrochlorothiazide(LNa)',
                      'C03AX': 'hydrochlorothiazide(LNa)',
                      'C09BX': 'hydrochlorothiazide(LNa)',
                      'M05BA': 'PamidronateZoledronate(HCa)',
                      'M05BB': 'etidronate(HCa)'}
        
        countf.replace(dict_lab).to_csv('ratio_'+loinc2+loinc3+'.csv')        
        
        plotdict = dict()
        
        tmps3 = tmps3.replace(dict_lab)
        tmps2 = tmps2.replace(dict_lab)
        
        tmps3 = tmps3.replace(dict_med)
        tmps2 = tmps2.replace(dict_med)        
        print('hello')
        for row in tmps3.iterrows():
            f1 = row[1][0]
            f2 = row[1][1]    
            tmps2X = tmps2[(tmps2['feature1']==f1) & (tmps2['feature2']==f2)]
            tmps2X  = tmps2X.reset_index(drop=True)
            fp.forestplot(tmps2X, 
                          estimate="rho",  # col containing estimated effect size 
                          ll="95ci_min", hl="95ci_max",  # columns containing conf. int. lower and higher limits
                          varlabel="site",  # column containing variable label
                          ylabel="Confidence interval",  # y-label title
                          xlabel=f1+' vs '+f2+ '('+dict_lab[loinc2]+')',
                          groupvar="range",
                          rightannote=["pvalue"],
                          right_annoteheaders=["pvalue"])
            plt.savefig(plot_prefix+f1+'_'+f2+'_'+dict_lab[loinc2]+'_'+dict_lab[loinc3]+".png", bbox_inches="tight")
            plotdict[(f1,f2,loinc2,loinc3)] = plt.gca()
        return plotdict

    def calculate_all3(self):
        dfmlr = pd.read_csv('mlr6_cofounder3.csv')

        dfmlr['rho'] = dfmlr['pdiff']
        dfmlr['95ci_low'] = dfmlr['CI95%down_adj']
        dfmlr['95ci_high'] = dfmlr['CI95%up_adj']
        dfmlr['target'] = dfmlr['covar']
        dfmlr['pvalue'] = (~dfmlr['significant']).astype(int).round(2)

        dfmlr = dfmlr[['feature1','feature2','rho', 'site','target','range','pvalue','95ci_low','95ci_high', 'n', 'n0']]

        def pmean(df):
            return scipy.stats.ttest_1samp(df, 0)[1]
        tmpsmeanstat = dfmlr[['feature1','feature2','rho','target','range']].groupby(['feature1', 'feature2','target','range']).agg([np.nanmean, np.nanstd, pmean]).reset_index()
        tmpsmeanstat.columns = ['feature1','feature2','target','range','rho','std', 'pvalue']
        tmpsmeanstat['95ci_low'] = tmpsmeanstat['rho']-2*tmpsmeanstat['std']
        tmpsmeanstat['95ci_high'] = tmpsmeanstat['rho']+2*tmpsmeanstat['std']
        tmpsmeanstat['site'] = 'MEAN'
        tmpsmeanstat = tmpsmeanstat.drop('std',axis=1)
        tmpsmeanstat['pvalue'] = tmpsmeanstat['pvalue'].round(2)

        tmps2 = pd.concat([dfmlr, tmpsmeanstat])
        tmps2['95ci_min'] = tmps2[['95ci_low','95ci_high']].min(axis=1)
        tmps2['95ci_max'] = tmps2[['95ci_low','95ci_high']].max(axis=1)
        tmps3  = tmps2[['feature1','feature2', 'target']].drop_duplicates()

        dict_lab = {'2157-6':'creatineKinase(HK)',
                  '1920-8':'AST(HK)',      
                  '2532-0':'LDH(HK)',
                  '4542-7':'Haptoglobin(HK)', 
                  '3084-1':'uricAcid(HK)',
                '2823-3':'potassium',
                '17861-6':'calcium',
                '2951-2':'sodium', 
                '2160-0':'sCr',
                'FLAG':'FLAG'}

        dict_med = {'A07DA': 'DiphenoxylateLoperamide(LK)',
                      'C03CA': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03CB': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03EB': 'FurosemideBumetanideTorsemide(LKLNa)',
                      'C03AA': 'hydrochlorothiazide(LNa)',
                      'C03AB': 'hydrochlorothiazide(LNa)',
                      'C03AH': 'chlorothiazide(LNa)',
                      'C03BA': 'chlorthalidone(LNa)',
                      'C03EA': 'hydrochlorothiazide(LNa)',
                      'C09DX': 'SacubitrilValsartan(HK)',
                      'C03DA': 'eplerenone(HK)',
                      'C03XA': 'tolvaptan(LNa)',
                      'C09XA': 'hydrochlorothiazide(LNa)',
                      'C09DX': 'hydrochlorothiazide(LNa)',
                      'C03AX': 'hydrochlorothiazide(LNa)',
                      'C09BX': 'hydrochlorothiazide(LNa)',
                      'M05BA': 'PamidronateZoledronate(HCa)',
                      'M05BB': 'etidronate(HCa)'}

        plotdict = dict()

        tmps3 = tmps3.replace(dict_lab)
        tmps2 = tmps2.replace(dict_lab)

        tmps3 = tmps3.replace(dict_med)
        tmps2 = tmps2.replace(dict_med)      

        tmps2

        loinc2 = '2823-3'
        loinc3 = 'FLAG'
        plot_prefix = "plot3_"
        for row in tmps3.iterrows():
            f1 = row[1][0]
            f2 = row[1][1]
            f3 = row[1][2]
            tmps2X = tmps2[(tmps2['feature1']==f1) & (tmps2['feature2']==f2) & (tmps2['target']==f3)]
            tmps2X  = tmps2X.reset_index(drop=True)
            fp.forestplot(tmps2X, 
                      estimate="rho",  # col containing estimated effect size 
                      ll="95ci_min", hl="95ci_max",  # columns containing conf. int. lower and higher limits
                      varlabel="site",  # column containing variable label
                      ylabel="Confidence interval",  # y-label title
                      xlabel=f1+' vs '+f2+ '('+f3+')',
                      groupvar="range",
                      rightannote=["pvalue", "n", "n0"],
                      right_annoteheaders=["pvalue", "n(cofounder)", "n(base)"])
            plt.savefig(plot_prefix+f1+'_'+f2+'_'+f3+"_.png", bbox_inches="tight")
            plotdict[(f1,f2,loinc2,loinc3)] = plt.gca()

        tmps2X = tmps2[(tmps2['feature1']==f1) & (tmps2['feature2']==f2) & (tmps2['target']==f3)]

        tmps2X  = tmps2X.reset_index(drop=True)

        fp.forestplot(tmps2X, 
                  estimate="rho",  # col containing estimated effect size 
                  ll="95ci_min", hl="95ci_max",  # columns containing conf. int. lower and higher limits
                  varlabel="site",  # column containing variable label
                  ylabel="Confidence interval",  # y-label title
                  xlabel=f1+' vs '+f2+ '('+f3+')',
                  groupvar="range",
                  rightannote=["pvalue", "n", "n0"],
                  right_annoteheaders=["pvalue", "n(cofounder)", "n(base)"])
        plt.savefig("plot3_"+f1+'_'+f2+'_'+dict_lab[loinc2]+'_'+dict_lab[loinc3]+".png", bbox_inches="tight")

    def combineimg(self):
        #Read the two images
        def combineimg_sub(f1,f2,outname):
            image1 = Image.open(f1)
            image2 = Image.open(f2)
            #resize, first image
            image2 = image2.resize(image1.size)
            image1_size = image1.size
            image2_size = image2.size
            new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
            new_image.paste(image1,(0,0))
            new_image.paste(image2,(image1_size[0],0))
            new_image.save(outname)

        files = [x for x in os.listdir() if 'plot2' in x]

        filesdf = pd.DataFrame(pd.DataFrame(files)[0].str.split('.').str[0].str.split('_').tolist(), columns= ['A','B','C','E','F'])

        filesdf['fname'] = filesdf['A']+'_'+filesdf['B']+'_'+filesdf['C']+'_'+filesdf['E']+'_'+filesdf['F']+'.png'

        for name, group in filesdf.groupby(['B','E','F']):
            try:
                outname = group['A'].iloc[0]+'c_'+group['B'].iloc[0]+'_'+group['E'].iloc[0]+'_'+group['F'].iloc[0]+'.png'
                if (group['C'].iloc[0] == 'FLAG') or (group['C'].iloc[0] == 'sCr'):
                    combineimg(group['fname'].iloc[1],group['fname'].iloc[0],outname)            
                else:
                    combineimg(group['fname'].iloc[0],group['fname'].iloc[1],outname)
            except:
                pass

    def combine_output(self):

        files = [x for x in os.listdir() if 'myco2_corr2' in x ]
        files = [x for x in files if 'FLAG' in x]

        files_u = [x for x in files if 'upper' in x ]
        files_l = [x for x in files if 'lower' in x ]
        files_f = [x for x in files if 'full' in x ]

        df_u = list()
        for file in files_u:
            tmp = pd.read_csv(file)
            df_u.append(tmp)
        df_u = pd.concat(df_u)
        df_u['range'] = 'upper'

        df_l = list()
        for file in files_l:
            tmp = pd.read_csv(file)
            df_l.append(tmp)
        df_l = pd.concat(df_l)
        df_l['range'] = 'lower'

        df_f = list()
        for file in files_f:
            tmp = pd.read_csv(file)
            df_f.append(tmp)
        df_f = pd.concat(df_f)
        df_f['range'] = 'full'

        df = pd.concat([df_u, df_l, df_f])
        #df = df[['ratio', 'Unnamed: 0', 'Coef._xy', 'P>|t|_xy', 'target', 'range', 'site']]
        df = df[['ratio', 'Unnamed: 0', 'Coef._xy', 'P>|z|_xy', 'target', 'range', 'site']]

        np.round(df[(abs(df['ratio'])>=0.1) & (df['P>|z|_xy']<=0.1)],6).to_csv('mlr6_cofounder2.csv')