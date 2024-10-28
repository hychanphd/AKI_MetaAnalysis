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

class mycorr:
    
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

    def copy(self, new_instance):
        self.site = new_instance.site
        self.labX = new_instance.labX.copy() if new_instance.labX is not None else None
        self.vital = new_instance.vital.copy() if new_instance.vital is not None else None
        self.vitalX = new_instance.vitalX.copy() if new_instance.vitalX is not None else None
        self.onset = new_instance.onset.copy() if new_instance.onset is not None else None
        self.medX = new_instance.medX.copy() if new_instance.medX is not None else None
        
        # self.loinc1 = new_instance.loinc1
        # self.loinc2 = new_instance.loinc2
        # self.loinc3 = new_instance.loinc3
    
    def get_labdata(self, site='UTHSCSA'):
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
    
    def get_vitaldata(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        self.vital = pd.read_csv(datafolder+site+'/raw/AKI_VITAL.csv')
        self.vital['MEASURE_DATE_TIME'] = pd.to_datetime(self.vital['MEASURE_DATE_TIME'])

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET', 'AKI2_ONSET', 'AKI3_ONSET']]
        x = x.bfill(axis=1)
        x = x[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']]
        x['AKI1_ONSET'] = pd.to_datetime(x['AKI1_ONSET'])        

        #        xxx = self.vitalxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.vital.merge(x, on=['PATID','ENCOUNTERID'], how='left')
        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.vitalX = xxx[xxx['AKI1_ONSET']-xxx['MEASURE_DATE_TIME']>=pd.Timedelta(1, "d")]
        #        self.vitalxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(0, "d")]
        self.vitalX = self.vitalX[self.vitalX['MEASURE_DATE_TIME']>=pd.to_datetime(self.vitalX['ADMIT_DATE'])]
    
    def get_onsetdata(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        onset = pd.read_csv(datafolder+site+'/raw/AKI_ONSETS.csv') 
        self.onset = onset[onset['NONAKI_SINCE_ADMIT'].isnull()]
    
    def get_meddata(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        self.med = pd.read_csv(datafolder+site+'/raw/AKI_AMED.csv')
        
        self.med['MEDADMIN_START_DATE_TIME'] = pd.to_datetime(self.med['MEDADMIN_START_DATE_TIME'])

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET', 'AKI2_ONSET', 'AKI3_ONSET']]
        x = x.bfill(axis=1)
        x = x[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']]
        x['AKI1_ONSET'] = pd.to_datetime(x['AKI1_ONSET'])        

        #        xxx = self.medxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.med.merge(x, on=['PATID','ENCOUNTERID'], how='left')
        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.medX = xxx[xxx['AKI1_ONSET']-xxx['MEDADMIN_START_DATE_TIME']>=pd.Timedelta(1, "d")]
        #        self.medxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE_TIME']>=pd.Timedelta(0, "d")]
        self.medX = self.medX[self.medX['MEDADMIN_START_DATE_TIME']>=pd.to_datetime(self.medX['ADMIT_DATE'])]
    
    def set_loincs_pair(self, loinc1 = None, loinc2 = None, loinc3 = None, vitalcode = None, medcode_rx = None, medcode_nd = None):
        self.loinc1 = loinc1
        self.loinc2 = loinc2
        self.loinc3 = loinc3
        self.vitalcode = vitalcode
        self.medcode_rx = medcode_rx
        self.medcode_nd = medcode_nd        
    
    def extract_lab(self):
        labx = self.labX[(self.labX['LAB_LOINC'].isin(self.loinc1+self.loinc2+self.loinc3))]
        labx = labx[['PATID','ENCOUNTERID','SPECIMEN_DATE_TIME','LAB_LOINC','RESULT_NUM']]
        labx = labx.groupby(['PATID','ENCOUNTERID','SPECIMEN_DATE_TIME','LAB_LOINC']).mean().reset_index()
        self.lablast = labx.sort_values('SPECIMEN_DATE_TIME').groupby(['PATID','ENCOUNTERID', 'LAB_LOINC']).last().reset_index()
        # lablast = lablast.drop('SPECIMEN_DATE_TIME',axis=1).pivot(index=['PATID','ENCOUNTERID'], columns = 'LAB_LOINC', values='RESULT_NUM')
        
    def extract_vital(self):
        vitalx = self.vitalX[['PATID','ENCOUNTERID', 'MEASURE_DATE_TIME', 'SYSTOLIC', 'DIASTOLIC']]
        vitalx_sys = vitalx.drop('DIASTOLIC',axis=1).sort_values('MEASURE_DATE_TIME').groupby(['PATID','ENCOUNTERID']).last()
        vitalx_dia = vitalx.drop('SYSTOLIC',axis=1).sort_values('MEASURE_DATE_TIME').groupby(['PATID','ENCOUNTERID']).last()
        self.vitallast = vitalx_sys.drop('MEASURE_DATE_TIME',axis=1).merge(vitalx_dia.drop('MEASURE_DATE_TIME',axis=1), left_index=True, right_index=True, how='outer').reset_index()
        
    def extract_med(self):
        # medcode_nd = {338011704: 'A', 338114803: 'B'}
        # medcode_rx = {338011704: 'A', 338114803: 'B'}
        medX2_nd = self.medX[(self.medX['MEDADMIN_TYPE']=='ND') & (self.medX['MEDADMIN_CODE'].isin(self.medcode_nd.keys()))]
        medX2_nd = medX2_nd.replace({"MEDADMIN_CODE":self.medcode_nd})
        medX2_rx = self.medX[(self.medX['MEDADMIN_TYPE']=='RX') & (self.medX['MEDADMIN_CODE'].isin(self.medcode_rx.keys()))]
        medX2_rx = medX2_rx.replace({"MEDADMIN_CODE":self.medcode_rx})
        medX2 = pd.concat([medX2_nd, medX2_rx])
        self.medlast = medX2[['PATID','ENCOUNTERID', 'MEDADMIN_CODE']].drop_duplicates()
        self.medlast['dummy'] = 1
        #medX2.pivot(index=['PATID','ENCOUNTERID'], columns = 'MEDADMIN_CODE', values = 'dummy').fillna(0).astype(int)
        
    def calculate_corr(self, datarange='full'):
        one_table = self.lablast.drop('SPECIMEN_DATE_TIME',axis=1).pivot(index=['PATID','ENCOUNTERID'], columns = 'LAB_LOINC', values='RESULT_NUM').reset_index().copy()
        try:
            one_table = one_table.merge(self.vitallast, on = ['PATID','ENCOUNTERID'], how='outer')
        except:
            pass
        try:
            medp = self.medlast.pivot(index=['PATID','ENCOUNTERID'], columns = 'MEDADMIN_CODE', values = 'dummy')
            one_table = one_table.merge(medp, on = ['PATID','ENCOUNTERID'], how='outer')
            one_table[medp.columns] = one_table[medp.columns].fillna(0)
        except:
            pass
        
        if datarange == 'upper':
            one_table = self.paitientlist_upper.merge(one_table, on = ['PATID','ENCOUNTERID'], how='inner')
        elif datarange == 'lower':
            one_table = self.paitientlist_lower.merge(one_table, on = ['PATID','ENCOUNTERID'], how='inner')
        
        self.one_table = one_table
        # one_table = self.vitallast.merge(self.medlast.pivot(index=['PATID','ENCOUNTERID'], columns = 'MEDADMIN_CODE', values = 'dummy'), on = ['PATID','ENCOUNTERID'], how='outer').merge(self.lablast.drop('SPECIMEN_DATE_TIME',axis=1).pivot(index=['PATID','ENCOUNTERID'], columns = 'LAB_LOINC', values='RESULT_NUM').reset_index()
        resdf = list()
        for x in self.loinc1+['SYSTOLIC', 'DIASTOLIC']+ list(np.unique(list(self.medcode_rx.values())+list(self.medcode_nd.values()))):
            for y in self.loinc2+self.loinc3:
                try:
                    xloinc = x
                    yloinc = y
                    xtmp = one_table[[xloinc,yloinc]].dropna()
                    res = scipy.stats.pearsonr(xtmp[xloinc], xtmp[yloinc])
                    resv = pd.DataFrame([[xloinc, yloinc, res.statistic, res.pvalue, res.confidence_interval(confidence_level=0.95).low, res.confidence_interval(confidence_level=0.95).high]], columns=['feature1', 'feature2', 'rho', 'pvalue', '95ci_low', '95ci_high'])
                    resdf.append(resv)
                except:
                    pass
        try:
            xloinc = self.loinc2[0]
            yloinc = self.loinc3[0]
            xtmp = one_table[[xloinc,yloinc]].dropna()
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
                rxy = sm.OLS(tmpone_table[y], X2).fit()
                Xz = tmpone_table[[x, z]]
                Xz2 = sm.add_constant(Xz)
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
        
    def set_range(self):
        if self.loinc2[0] == '2823-3':
            threshold_lower = 3.2
            threshold_upper = 4.4
        elif self.loinc2[0] == '2951-2':
            threshold_lower = 132
            threshold_upper = 144            
        elif self.loinc2[0] == '17861-6':
            threshold_lower = 8
            threshold_upper = 9.25                        

        self.paitientlist_upper = self.lablast[(self.lablast['LAB_LOINC']==self.loinc2[0]) & (self.lablast['RESULT_NUM']>=threshold_upper)][['PATID','ENCOUNTERID']].drop_duplicates()
        self.paitientlist_lower = self.lablast[(self.lablast['LAB_LOINC']==self.loinc2[0]) & (self.lablast['RESULT_NUM']<=threshold_lower)][['PATID','ENCOUNTERID']].drop_duplicates()
            
    def calculate_site(self, site):
        # import mycorr
        # importlib.reload(mycorr)
        myco = mycorr()
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        with open(datafolder+'myco_tmp_'+site+'.pkl', "rb") as f:
            myco2 = pickle.load(f)
        myco.copy(myco2)

        loinc1 = ['2157-6', #creatine kinase
                  '1920-8', #AST      
                  '2532-0', #LDH
                  '4542-7', #Haptoglobin 
                  '3084-1'] #uric acid

#        loinc2 = ['2823-3'] # potassium
        loinc2 = ['17861-6'] # calcium
#        loinc2 = ['2951-2'] # sodium       
        
        loinc3 = ['2160-0'] #sCr
        # medcode_nd = {'338011704': 'A', '338114803': 'B'}
        # medcode_rx = {'338011704': 'A', '338114803': 'B'}
        drugs = pd.read_csv('druglist.csv').drop('Unnamed: 0',axis=1)
        drugs = drugs[drugs['target'] == loinc2[0]]
        rxdrugs = drugs[drugs['type']=='rx'][['code', 'name']]
        medcode_rx = rxdrugs.set_index('code')['name'].to_dict()
        ndcdrugs = drugs[drugs['type']=='ndc'][['code', 'name']]
        medcode_nd = ndcdrugs.set_index('code')['name'].to_dict()        
        myco.set_loincs_pair(loinc1=loinc1, loinc2=loinc2, loinc3=loinc3, medcode_rx=medcode_rx, medcode_nd=medcode_nd)
        myco.extract_lab()
        myco.extract_vital()
        myco.extract_med()
        myco.set_range()
        datarange = 'full'
        myco.calculate_corr(datarange=datarange)
        myco.resdf['site'] = site        
        myco.resdf['target'] = myco.loinc2[0]                
        myco.resdf.to_csv('myco_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')   
        try:
            myco.calculate_corr2()
            myco.res_df2['site'] = site        
            myco.res_df2['target'] = myco.loinc2[0]                
            myco.res_df2.to_csv('myco_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')   
        except:
            pass
        
        datarange = 'upper'
        myco.calculate_corr(datarange=datarange)
        myco.resdf['site'] = site
        myco.resdf['target'] = myco.loinc2[0]                        
        myco.resdf.to_csv('myco_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')        
        try:        
            myco.calculate_corr2()
            myco.res_df2['site'] = site        
            myco.res_df2['target'] = myco.loinc2[0]                
            myco.res_df2.to_csv('myco_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')           
        except:
            pass
        
        datarange = 'lower'
        myco.calculate_corr(datarange=datarange)
        myco.resdf['site'] = site
        myco.resdf['target'] = myco.loinc2[0]                        
        myco.resdf.to_csv('myco_corr_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')      
        try:
            myco.calculate_corr2()
            myco.res_df2['site'] = site        
            myco.res_df2['target'] = myco.loinc2[0]                
            myco.res_df2.to_csv('myco_corr2_'+datarange+'_'+site+'_'+myco.loinc2[0]+'.csv')   
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
            
    def calculate_all2(self, loinc2='2823-3'):
        def cal_temp_stat(datarange):
            sites = ['MCW', 'UMHC', 'UNMC', 'UofU', 'UTHSCSA', 'KUMC', 'UTSW', 'UIOWA', 'UPITT', 'IUR']
            tmps = list()
            for site in sites:
                try:
                    tmp = pd.read_csv('myco_corr_'+datarange+'_'+site+'_'+loinc2+'.csv')
                    tmps.append(tmp)
                except:
                    pass
            tmps = pd.concat(tmps)
            tmps = tmps.drop('Unnamed: 0',axis=1)

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
        dict_lab = {'2157-6':'creatine kinase',
                  '1920-8':'AST',      
                  '2532-0':'LDH',
                  '4542-7':'Haptoglobin', 
                  '3084-1':'uric acid',
                '2823-3':'potassium',
                '17861-6':'calcium',
                '2951-2':'sodium', 
                '2160-0':'sCr'}
        countf.replace(dict_lab).to_csv('ratio_'+loinc2+'.csv')        
        
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
                          xlabel=f1+' vs '+f2+ '('+loinc2+')',
                          pval="pvalue",
                          groupvar="range")
            plt.savefig("plot_"+f1+'_'+f2+'_'+loinc2+".png", bbox_inches="tight")
            
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
        
