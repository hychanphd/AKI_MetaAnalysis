import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def read_AKI_ONSETS(ct_names, raw_path, fill_in_rec_num_df = False):
    
    AKI_ONSETS_dfs = dict()
    
    for ct in ct_names:
        print('\n' + ct + ':')
        data_path = raw_path + ct + '/raw/'
        if (ct == 'UPITT') or (ct == 'UTHSCSA') or (ct == 'UIOWA') or (ct == 'UNMC'):
            AKI_onset = pd.read_csv(data_path + "AKI_ONSETS.csv", delimiter = ',')
        elif (ct == 'UTSW'):
            AKI_onset = pd.read_csv(data_path + "AKI_ONSETS.dsv", delimiter = '|')
        elif (ct == 'MCW'):
            AKI_onset = pd.read_csv(data_path + "AKI_ONSETS.dsv", delimiter = '|')
            Upper_Case_Columns(AKI_onset)
        elif (ct == 'UMHC'):
            AKI_onset = pd.read_csv(data_path + "DEID_AKI_ONSETS.csv", delimiter = ',')
        elif (ct == 'UofU'):
            AKI_onset = pd.read_csv(data_path + "AKI_ONSETS.csv", delimiter = '|')
        elif (ct == 'KUMC'):
            AKI_onset = pd.read_csv(data_path + "AKI_ONSETS.csv", delimiter = ',')
            AKI_onset_cols = AKI_onset.columns.tolist()
            # AKI_onset_cols = [s[:-len('"+PD.DATE_SHIFT"')] \
            #                   if s.endswith('"+PD.DATE_SHIFT"') else s for s in AKI_onset_cols]
            AKI_onset.columns = AKI_onset_cols
        
        AKI_onset.rename(columns={'ENCOUNTERID': 'ONSETS_ENCOUNTERID'}, inplace = True) 
        
        AKI_ONSETS_dfs[ct] = AKI_onset
        print('Initially, there are %d encounters in total!' %(len(AKI_ONSETS_dfs[ct].ONSETS_ENCOUNTERID.unique())))
        
        if fill_in_rec_num_df:
            records_num_df.loc['Total number of encounters', ct] = len(AKI_ONSETS_dfs[ct].ONSETS_ENCOUNTERID.unique())
            
    return AKI_ONSETS_dfs

#read Scr records, here we kept the historical records(DAYS_SINCE_ADMIT < 0)
def read_AKI_LAB_SCR(ct_names, raw_path):
    SCR_dfs = dict()
    use_cols = ['ONSETS_ENCOUNTERID','PATID','ENCOUNTERID','SPECIMEN_DATE','RESULT_NUM', 'DAYS_SINCE_ADMIT']

    for ct in ct_names:
        data_path = raw_path + ct + '/raw/'
        if (ct == 'UPITT') or (ct == 'UTHSCSA') or (ct == 'UIOWA'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = ',', usecols=use_cols)
        elif (ct == 'UTSW'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.dsv", delimiter = '|', usecols=use_cols)
        elif (ct == 'MCW'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.dsv", delimiter = '|', usecols=list(map(str.lower, use_cols)))
            Upper_Case_Columns(SCR_df)
        elif (ct == 'UMHC'):
            SCR_df = pd.read_csv(data_path + "DEID_AKI_LAB_SCR.csv", delimiter = ',', usecols=use_cols)
        elif (ct == 'UofU'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = '|', usecols=use_cols)
        elif (ct == 'KUMC_ORCALE'):
            use_cols = ['ONSETS_ENCOUNTERID','PATID','ENCOUNTERID',
                        'SPECIMEN_DATE"+PD.DATE_SHIFT"','RESULT_NUM', 'DAYS_SINCE_ADMIT']
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = ',', usecols=use_cols)
            SCR_df.columns = ['ONSETS_ENCOUNTERID','PATID','ENCOUNTERID', 'SPECIMEN_DATE','RESULT_NUM', 
                              'DAYS_SINCE_ADMIT']

        SCR_dfs[ct] = SCR_df
        
    return SCR_dfs

#read patients' demographical data
def read_AKI_DEMO(ct_names, raw_path):
    AKI_DEMO_dfs = dict()
    use_cols = ['ONSETS_ENCOUNTERID', 'AGE', 'PATID', 'SEX', 'RACE']

    for ct in ct_names:
        data_path = raw_path + ct + '/raw/'
        if (ct == 'UPITT') or (ct == 'UTHSCSA') or (ct == 'UIOWA') or (ct == 'KUMC_ORCALE'):
             AKI_DEMO = pd.read_csv(data_path + "AKI_DEMO.csv", delimiter = ',', usecols = use_cols)
        elif (ct == 'UTSW'):
            AKI_DEMO = pd.read_csv(data_path + "AKI_DEMO.dsv", delimiter = '|', usecols = use_cols)
        elif (ct == 'MCW'):
            AKI_DEMO = pd.read_csv(data_path + "AKI_DEMO.dsv", delimiter = '|', usecols = list(map(str.lower, use_cols)))
            Upper_Case_Columns(AKI_DEMO)
        elif (ct == 'UMHC'):
            AKI_DEMO = pd.read_csv(data_path + "DEID_AKI_DEMO.csv", delimiter = ',', usecols = use_cols)
        elif (ct == 'UofU'):
            AKI_DEMO = pd.read_csv(data_path + "AKI_DEMO.csv", delimiter = '|', 
                                           header=None, skiprows = 1, usecols=[0, 1, 2, 5, 17])
            AKI_DEMO.columns = use_cols
    
        AKI_DEMO_dfs[ct] = AKI_DEMO
        
    return AKI_DEMO_dfs

#read patients' diagnosis data
#cneters do not have a DX_DATE: UTHSCSA, UTSW, UofU
def read_AKI_DX(ct_names, raw_path):
    AKI_DX_dfs = dict()
    use_cols = ['PATID', 'DX_DATE', 'DX', 'DX_TYPE', 'DAYS_SINCE_ADMIT']
    ct_missing_DX_DATE = ['UTHSCSA', 'UTSW', 'UofU']
    
    for ct in ct_names:
        data_path = raw_path + ct + '/raw/'
        if (ct == 'UPITT') or (ct == 'UTHSCSA') or (ct == 'UIOWA'):
            AKI_DX = pd.read_csv(data_path + "AKI_DX.csv", delimiter = ',', usecols=use_cols)
            #adjust the col order of UIOWA
            if ct == 'UIOWA':
                AKI_DX = AKI_DX[use_cols]
        elif (ct == 'UTSW'):
            AKI_DX = pd.read_csv(data_path + "AKI_DX.dsv", delimiter = '|', usecols=use_cols)
        elif (ct == 'MCW'):
            AKI_DX = pd.read_csv(data_path + "AKI_DX.dsv", delimiter = '|', usecols=list(map(str.lower, use_cols)))
            Upper_Case_Columns(AKI_DX)
        elif (ct == 'UMHC'):
            AKI_DX = pd.read_csv(data_path + "DEID_AKI_DX.csv", delimiter = ',', usecols=use_cols)
        elif (ct == 'UofU'):
            AKI_DX = pd.read_csv(data_path + "AKI_DX.csv", delimiter = '|', header=None, 
                                           skiprows = 1, usecols=[2, 6, 8, 9, 20])
            AKI_DX.columns = use_cols
        elif (ct == 'KUMC_ORCALE'):
            AKI_DX = pd.read_csv(data_path + "AKI_DX.csv", delimiter = ',', 
                                 usecols=['PATID', 'DX_DATE"+PD.DATE_SHIFT"', 'DX', 
                                          'DX_TYPE', 'DAYS_SINCE_ADMIT'])
            AKI_DX.columns = use_cols
        
        if ct not in ct_missing_DX_DATE:
            AKI_DX['DX_DATE'] = pd.to_datetime(AKI_DX['DX_DATE'], format = 'mixed')
            AKI_DX['DX_DATE'] = AKI_DX['DX_DATE'].dt.strftime('%Y-%m-%d')
            AKI_DX['DX_DATE'] = pd.to_datetime(AKI_DX['DX_DATE'], format = 'mixed')
            
        AKI_DX_dfs[ct] = AKI_DX
        
    return AKI_DX_dfs