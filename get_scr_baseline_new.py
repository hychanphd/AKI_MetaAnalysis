def get_scr_baseline_new(df_scr, df_admit, dx, demo, c7day = 'MOST_RECENT', c365day = 'AVERAGE', cckd = 'DROP'):
    cohort_table = dict()
    
    # load & process dx data
#    dx = pd.read_pickle(filepath_lst[0]+'AKI_DX.pkl')  
    dx = dx[['PATID', 'ENCOUNTERID', 'DX', 'DX_DATE', 'DX_TYPE']] 
    dx = df_admit[['PATID', 'ENCOUNTERID', 'ADMIT_DATE']].merge(dx, on = ['PATID', 'ENCOUNTERID'], how = 'inner')
    dx['DAYS_SINCE_ADMIT'] = (dx['DX_DATE']-dx['ADMIT_DATE']).dt.days

    dx['DX'] = dx['DX'].astype(str)
    dx['DX_TYPE'] = dx['DX_TYPE'].astype(str)
    dx['DX_TYPE'] = dx['DX_TYPE'].replace('09', '9')
    
    # load & process demo data
#    demo = pd.read_pickle(filepath_lst[0]+'AKI_DEMO'+'.pkl')  
#    demo['MALE'] = demo['SEX'] == 'M'

#    demo['RACE_WHITE'] = demo['RACE'] == '05'
#    demo['RACE_BLACK'] = demo['RACE'] == '03'
    demo = demo[['PATID', 'ENCOUNTERID', 'AGE', 'MALE', 'RACE_WHITE', 'RACE_BLACK']]
    demo = demo.drop_duplicates()
    
    # estimate SCr Baseline
    pat_id_cols = ['PATID', 'ENCOUNTERID']
    complete_df = df_scr[['ENCOUNTERID', 'PATID', 'ADMIT_DATE', 'SPECIMEN_DATE', 'RESULT_NUM']]
 
    # 1. min between the min of 1-week prior admission SCr and within 24 hour after admission SCr
    # SCr within 24 hour after admission, that is admission day and one day after, get mean
    admission_SCr = complete_df[(complete_df.SPECIMEN_DATE >= complete_df.ADMIT_DATE) & \
                                (complete_df.SPECIMEN_DATE <= (complete_df.ADMIT_DATE + pd.Timedelta(days=1)))].copy()

    # Admission SCr is the mean of all the SCr within 24h admission
    admission_SCr = admission_SCr.groupby(pat_id_cols)['RESULT_NUM'].mean().reset_index()

    admission_SCr.rename(columns = {'RESULT_NUM': 'ADMISSION_SCR'}, inplace = True)

    # merge the ADMISSION_SCR back to the main frame
    complete_df = complete_df.merge(admission_SCr, 
                                    on = pat_id_cols,
                                    how = 'left')

    # SCr within 7 days prior to admission
    one_week_prior_admission = complete_df[(complete_df.SPECIMEN_DATE >= complete_df.ADMIT_DATE - pd.Timedelta(days=7)) & \
                                           (complete_df.SPECIMEN_DATE < complete_df.ADMIT_DATE)].copy()
    one_week_prior_admission = one_week_prior_admission.sort_values(by = ['PATID', 'ENCOUNTERID','SPECIMEN_DATE'])
    
    if c7day == 'MOST_RECENT':
        one_week_prior_admission = one_week_prior_admission.groupby(pat_id_cols)['RESULT_NUM'].last().reset_index()
    else:
        one_week_prior_admission = one_week_prior_admission.groupby(pat_id_cols)["RESULT_NUM"].min().reset_index()
        
    one_week_prior_admission = one_week_prior_admission.rename(columns = {'RESULT_NUM': 'ONE_WEEK_SCR'})

    complete_df = complete_df.merge(one_week_prior_admission, 
                                    on = pat_id_cols,
                                    how = 'left')

    # take the min between one week SCr and admission SCr
    complete_df.loc[complete_df.ONE_WEEK_SCR.notna(), 'BASELINE_EST_1'] = \
                np.min(complete_df.loc[complete_df.ONE_WEEK_SCR.notna(), ['ONE_WEEK_SCR','ADMISSION_SCR']], axis = 1)

    complete_dfe = complete_df.drop(['SPECIMEN_DATE', 'RESULT_NUM'],axis=1).drop_duplicates()
    cohort_table['ONE_WEEK_SCR_YES'] = complete_dfe.ONE_WEEK_SCR.notna().sum()
    cohort_table['ONE_WEEK_SCR_NO'] = complete_dfe.ONE_WEEK_SCR.isna().sum()    
    cohort_table['ONE_WEEK_SCR_ONE_WEEK_SCR'] = (complete_dfe.ONE_WEEK_SCR.notna() & (complete_dfe['ONE_WEEK_SCR']==complete_dfe['BASELINE_EST_1'])).sum()
    cohort_table['ONE_WEEK_SCR_ADMISSION_SCR'] = (complete_dfe.ONE_WEEK_SCR.notna() & (complete_dfe['ONE_WEEK_SCR']!=complete_dfe['BASELINE_EST_1'])).sum()
        
    ori_num_unique_combinations = df_scr.groupby(['PATID', 'ENCOUNTERID']).ngroups
    # get the percentage of encounters that do not have past 7-day records
    criterion1_no_missing = complete_df.loc[complete_df.ONE_WEEK_SCR.notna(), :].groupby(pat_id_cols).ngroups
    criterion1_missing_rate = 1 - criterion1_no_missing / ori_num_unique_combinations

    # 2. pre-admission 365-7 day mean
    # here we only care about SCr measurements within 1 year before hospitalization
    one_year_prior_admission = complete_df[(complete_df.SPECIMEN_DATE < (complete_df.ADMIT_DATE - pd.Timedelta(days=7))) & \
                                     (complete_df.SPECIMEN_DATE >= (complete_df.ADMIT_DATE - pd.Timedelta(days=365.25)))].copy()
    one_year_prior_admission = one_year_prior_admission.sort_values(by = ['PATID', 'ENCOUNTERID','SPECIMEN_DATE'])
    one_year_prior_admission = one_year_prior_admission.loc[:, pat_id_cols + ['RESULT_NUM']]
    
    if c365day == 'AVERAGE':
        one_year_prior_admission = one_year_prior_admission.groupby(pat_id_cols)['RESULT_NUM'].mean().reset_index()
    else:
        one_year_prior_admission = one_year_prior_admission.groupby(pat_id_cols)['RESULT_NUM'].last().reset_index()  # or mean()
    
    one_year_prior_admission.rename(columns = {'RESULT_NUM': 'ONE_YEAR_SCR'}, inplace = True)
    
    complete_df = complete_df.merge(one_year_prior_admission, 
                                    on = pat_id_cols,
                                    how = 'left')
    
    # take the min between one week SCr and admission SCr
    complete_df.loc[complete_df.ONE_YEAR_SCR.notna(), 'BASELINE_EST_2'] = \
                np.min(complete_df.loc[complete_df.ONE_YEAR_SCR.notna(), ['ONE_YEAR_SCR', 'ADMISSION_SCR']], axis = 1)

    # priority 1: 7day SCr, priority 2: one year SCr
    complete_df['BASELINE_NO_INVERT'] = \
                np.where(complete_df['BASELINE_EST_1'].isna(), complete_df['BASELINE_EST_2'], complete_df['BASELINE_EST_1'])

    complete_dfe = complete_df.drop(['SPECIMEN_DATE', 'RESULT_NUM'],axis=1).drop_duplicates()
    cohort_table['ONE_YEAR_SCR_YES'] = (complete_dfe.ONE_WEEK_SCR.isna() & complete_dfe.ONE_YEAR_SCR.notna()).sum()
    cohort_table['ONE_YEAR_SCR_NO'] = (complete_dfe.ONE_WEEK_SCR.isna() & complete_dfe.ONE_YEAR_SCR.isna()).sum()
    cohort_table['ONE_YEAR_SCR_ONE_WEEK_SCR'] = (complete_dfe.ONE_WEEK_SCR.isna() & complete_dfe.ONE_YEAR_SCR.notna() & (complete_dfe['ONE_YEAR_SCR']==complete_dfe['BASELINE_EST_2'])).sum()
    cohort_table['ONE_YEAR_SCR_ADMISSION_SCR'] = (complete_dfe.ONE_WEEK_SCR.isna() & complete_dfe.ONE_YEAR_SCR.notna() & (complete_dfe['ONE_YEAR_SCR']!=complete_dfe['BASELINE_EST_2'])).sum()    
    
    # 3. Invert CKD-EPI (2021) to estimate baseline (only for non-CKD patients)
    # get those encounters for which we need to impute baseline
    pat_to_invert = complete_df.loc[complete_df.BASELINE_NO_INVERT.isna(), pat_id_cols+['ADMIT_DATE', 'ADMISSION_SCR']]
    # one patient one row
    pat_to_invert.drop_duplicates(subset=pat_id_cols, keep='first', inplace = True)


    pat_dx = pat_to_invert.merge(dx.drop(['ENCOUNTERID', 'ADMIT_DATE'], axis = 1), 
                                              on = 'PATID', 
                                              how = 'left')

    # calculate DX_DATE when it is missing
    pat_dx.loc[pat_dx.DX_DATE.isna(), 'DX_DATE'] = \
            pat_dx.loc[pat_dx.DX_DATE.isna(), 'ADMIT_DATE'] + \
            pd.to_timedelta(pat_dx.loc[pat_dx.DX_DATE.isna(), 'DAYS_SINCE_ADMIT'], unit='D')

    # check patients that do not have DX in the database
    #pat_dx.DX_DATE.isna().mean()

    # filter out those DX after admission
    pat_dx = pat_dx[pat_dx.DX_DATE <= pat_dx.ADMIT_DATE]

    # get the default eGFR for inversion: default to 75 for non-CKD patients, average of eGFR from staging criteria for CKD patients
    pat_dx['DFLT_eGFR'] = 75

    pat_dx.loc[pat_dx['DX'].isin(['585.3', 'N18.3']), 'DFLT_eGFR'] = 90/2
    pat_dx.loc[pat_dx['DX'].isin(['585.4', 'N18.4']), 'DFLT_eGFR'] = 45/2
    pat_dx.loc[pat_dx['DX'].isin(['585.5', 'N18.5']), 'DFLT_eGFR'] = 15/2
#    pat_dx.loc[pat_dx['DX'].isin(['585.6', 'N18.6']), 'DFLT_eGFR'] = 15/2

    pat_def_egfr = pat_dx.groupby(pat_id_cols)['DFLT_eGFR'].min().reset_index()

    cohort_table['MDRD_NOCKD'] = (pat_def_egfr['DFLT_eGFR'] == 75).sum()
    cohort_table['MDRD_CKD3']  = (pat_def_egfr['DFLT_eGFR'] == 90/2).sum()
    cohort_table['MDRD_CKD4']  = (pat_def_egfr['DFLT_eGFR'] == 45/2).sum()
    cohort_table['MDRD_CKD5']  = (pat_def_egfr['DFLT_eGFR'] == 15/2).sum()
        
    pat_to_invert= pat_to_invert.merge(pat_def_egfr, on = pat_id_cols, how = 'left')
    pat_to_invert['DFLT_eGFR'] = pat_to_invert['DFLT_eGFR'].fillna(75)

    pat_to_invert['DROPCKD'] = pat_to_invert['DFLT_eGFR'] != 75
    
    #pat_to_invert.DFLT_eGFR.value_counts()

    # Backcalculation for patients
    # merge DEMO with pat_to_invert
    pat_to_invert = pat_to_invert.merge(demo, on = pat_id_cols, how = 'left')
    
    KDIGO_baseline = np.array([
        [1.5, 1.3, 1.2, 1.0],
        [1.5, 1.2, 1.1, 1.0],
        [1.4, 1.2, 1.1, 0.9],
        [1.3, 1.1, 1.0, 0.9],
        [1.3, 1.1, 1.0, 0.8],
        [1.2, 1.0, 0.9, 0.8]
    ])
    KDIGO_baseline = pd.DataFrame(KDIGO_baseline, columns = ["Black males", "Other males",
                                                            "Black females", "Other females"],
                                 index = ["20-24", "25-29", "30-39", "40-54", "55-65", ">65"])    
    
    
    # estimate SCr from eGFR
    pat_to_invert.loc[:, 'BASELINE_INVERT'] = pat_to_invert.apply(inverse_MDRD, args = (KDIGO_baseline,), axis = 1) #pat_to_invert.apply(inverse_CKDEPI21, axis = 1)

    # take minimum of inverted SCr and admission SCr
    pat_to_invert['BASELINE_EST_3'] = np.min(pat_to_invert[['ADMISSION_SCR', 'BASELINE_INVERT', 'DROPCKD']], axis = 1)

    # merge back the computation results
    complete_df = complete_df.merge(pat_to_invert[pat_id_cols + ['BASELINE_EST_3', 'DROPCKD']], 
                                    on = pat_id_cols,
                                    how = 'left')

    # replace the old baseline
    complete_df['SERUM_CREAT_BASE'] = np.min(complete_df[['BASELINE_NO_INVERT', 'BASELINE_EST_3']], axis = 1)

    if cckd:
        complete_df = complete_df[complete_df['DROPCKD'] & complete_df['BASELINE_NO_INVERT'].isna()]
    complete_df = complete_df.drop('DROPCKD', axis=1)
        
    # drop those still cannot find baseline
    complete_df = complete_df.dropna(subset=['SERUM_CREAT_BASE'])

    return complete_df.drop_duplicates(), cohort_table