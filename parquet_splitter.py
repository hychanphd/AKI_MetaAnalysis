import utils_function
import pandas as pd
import time

def spliter(site, dataname):
    suffix=''
    if dataname == 'vital_old_nooutliner':
        dataname = 'vital_old'
        suffix = '_nooutliner'
    
    if dataname == 'lab_g_nooutliner':
        dataname = 'lab_g'
        suffix = '_nooutliner'               

    
    print(f"Splitting p0_{dataname}_{site}{suffix}.parquet")
    
    configs_variable = utils_function.read_config(site)
    site, datafolder, home_directory = utils_function.get_commons(configs_variable)
    onset = pd.read_parquet(configs_variable['datafolder']+configs_variable['site']+'/p0_onset_'+configs_variable['site']+'.parquet')
    onset['PATID'] = onset['PATID'].astype(str)
    onset['ENCOUNTERID'] = onset['ENCOUNTERID'].astype(str)
    years = list(pd.to_datetime(onset['ADMIT_DATE']).dt.year.unique())    
    
    df = pd.read_parquet(datafolder+site+ f"/p0_{dataname}_{site}{suffix}.parquet")
    df['PATID'] = df['PATID'].astype(str)
    df['ENCOUNTERID'] = df['ENCOUNTERID'].astype(str)

    for year in years:
        print(f"Splitting p0_{dataname}_{site}_{year}{suffix}.parquet")
        newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')[['PATID', 'ENCOUNTERID']]
        newdfX['PATID'] = newdfX['PATID'].astype(str)
        newdfX['ENCOUNTERID'] = newdfX['ENCOUNTERID'].astype(str)

        newdfX = newdfX.merge(df, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left')
        newdfX.to_parquet(datafolder+site+ f"/p0_{dataname}_{site}_{year}{suffix}.parquet")
        
if __name__ == "__main__":
    site_list = ['KUMC', 'UTSW', 'MCW', 'UofU', 'UIOWA', 'UMHC', 'UPITT', 'UTHSCSA', 'UNMC']
#    datanames = ['demo', 'vital_old', 'vital_old_nooutliner', 'dx', 'px', 'lab_g', 'lab_g_nooutliner', 'amed']

#    site_list = ['KUMC', 'UPITT']
    datanames = ['lab_g', 'lab_g_nooutliner']


    # for site in site_list:
    #     for dataname in datanames:
    #         spliter(site, dataname)
    #         if dataname == 'vital_old' or dataname == 'lab_g':
    #             spliter(site, dataname, '_nooutliner')

    def runner(runner_wrapper, site, dataname):
        tic = time.perf_counter() 
        runner_wrapper(site, dataname)
        print(f"{site}:{dataname} finished in {toc - tic:0.4f} seconds")        
    
    para_list_local = [(runner, site, dataname) for dataname in datanames for runner in [spliter] for site in site_list]
    
    import multiprocessing as mp
    from pebble import ProcessPool
    from concurrent.futures import TimeoutError
    def task_done(future):
        try:
            result = future.result()  # blocks until results are ready
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised " + "error" +"\n"+error.traceback)

    with ProcessPool(max_workers=2) as pool:
        for paras in para_list_local:
            future = pool.schedule(runner, args=paras, timeout=86400)
            future.add_done_callback(task_done)

    print('done')
