library(arrow)
library(dplyr)
library(mgcv)
library(tidyverse)
library(data.table)
library(metagam)
library(latex2exp)
library(eivtools)
library(plotly)
library(stats)
library(plotrix)
library(fishmethods)
library(metafor)
library(jsonlite)
library(stringr)
library(doParallel)
library(parallel)
library(MASS)

stg <- "stg01"
fs <-  'nofs'
oversample <- 'raw'
model_type <- 'catd'

path <- '/home/hoyinchan/blue/Data/data2022/shapalltmp.parquet'
dfraw <- arrow::read_parquet(path)
dfraw <- dfraw %>% dplyr::select(-'__index_level_0__')
dfraw <- dfraw %>% dplyr::filter(site_d != 'MCRI') %>% dplyr::filter(site_m != 'MCRI')
dfraw$site_d <- as.factor(dfraw$site_d)
dfraw$site_m <- as.factor(dfraw$site_m)

targets <- unique(dfraw$Feature)

#cattarget <- list("PX:CH:J1940", "PX:09:96.72")
cattarget <- names(df)[sapply(df, is.logical)]

fit_proc <- function(eqn, dfraw2, target, type){
    xfit <- bam(eqn, data=dfraw2, method='REML')        
    print(summary(xfit))
    flush.console()
    sxfit<-summary(xfit)
    pxfit<-plot(xfit)
    for (i in 1:length(pxfit)){
        pxfit[[i]]$raw=NULL    
    }
    return(list(target, type, sxfit, pxfit))
}    

gam_proc <- function(outputname, cattarget, targets, dfraw) {
    resultlist <- list()
    for (target in targets){
        print(target)
        flush.console()
        dfraw2 <- dfraw %>% filter(Feature==target)            
        if (target %in% cattarget){
            dfraw2$Name <- as.factor(dfraw2$Name)
            eqnl <- val ~ Name + s(site_d,bs="re") + roc2
            eqnq <- val ~ Name + s(site_d,bs="re") + roc2
            eqns <- val ~ Name + s(site_d,bs="re") + roc2
            eqnt <- val ~ Name + s(site_d,bs="re") + roc2 
            resultl<-fit_proc(eqnl, dfraw2, target, 'linear')
            resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic')
            results<-fit_proc(eqns, dfraw2, target, 'spline')
            resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction')            
            result<-list(resultl, resultq, results, resultt)            
        } else {
            eqnl <- val ~ poly(Name,1) + s(site_d,bs="re") + roc2
            eqnq <- val ~ poly(Name,2) + s(site_d,bs="re") + roc2
            eqns <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re") + roc2
            eqnt <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re") + roc2 + ti(Name,roc2,bs='cr')            
            resultl<-fit_proc(eqnl, dfraw2, target, 'linear')
            resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic')
            results<-fit_proc(eqns, dfraw2, target, 'spline')
            resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction')            
            result<-list(resultl, resultq, results, resultt)
        }
        resultlist <- append(resultlist, list(result))
    }
    output_to_python <- toJSON(resultlist, force = TRUE)
    write(output_to_python, paste0(outputname))
}

target_combo = combn(targets,2, simplify = FALSE)
outputname <- "gamalltmp_double_interaction.json"

gam_proc2d <- function(cattarget, dfraw, f1, f2, stg, fs, oversample, model_type, returnf = FALSE) {

    f1str <- str_replace_all(f1,'::','_')
    f1str <- str_replace_all(f1str,'/','per')
    f1str <- str_replace_all(f1str,'\\(','_')
    f1str <- str_replace_all(f1str,'\\)','_')
    
    f2str <- str_replace_all(f2,'::','_')
    f2str <- str_replace_all(f2str,'/','per')
    f2str <- str_replace_all(f2str,'\\(','_')
    f2str <- str_replace_all(f2str,'\\)','_')    

    if (!returnf){
        filename <- paste0('/home/hoyinchan/blue/program_data/AKI_CDM_PY/MetaRegression/gam2d_tmp/','gam2d_tmp','_',f1str,'_',f2str,'_',stg,'_',fs,'_',oversample,'_',model_type,'.json')
        if (file.exists(filename)){
            return()
        }
    }
    
    eqn_cc <- val ~ s(Name.x,k=10,bs='cr') + s(Name.y,k=10,bs='cr') + s(site_d,bs="re") + roc2 + ti(Name.x,Name.y,k=10,bs='cr')
    eqn_cd <- val ~ s(Name.x,k=10,bs='cr') + s(Name.x,by=Name.y,k=10,bs='cr') + Name.y + s(site_d,bs="re") + roc2

    if (f1 %in% cattarget & !f2 %in% cattarget){
        tmp = f1
        f1 = f2
        f2 = tmp
    }

    dfraw21 <- dfraw %>% filter(Feature==f1)
    dfraw22 <- dfraw %>% filter(Feature==f2)
    dfraw23 <- dfraw21 %>% inner_join(dfraw22, by=c('ID','site_d', 'site_m', 'roc', 'roc2'))
    dfraw23 <- dfraw23 %>% mutate(val=val.x+val.y)

    if (!f1 %in% cattarget & f2 %in% cattarget){
        eqn <- eqn_cd
        dfraw23$Name.y <- as.factor(dfraw23$Name.y)
    } else if (!f1 %in% cattarget & !f2 %in% cattarget){
        eqn <- eqn_cc
    } else {
        return()
    }

    xfit <- bam(eqn, data=dfraw23, method='REML') 
    sxfit<-summary(xfit)
    pxfit<-plot(xfit)
    for (i in 1:length(pxfit)){
        pxfit[[i]]$raw=NULL
    }
    result<-list(f1, f2, as.list(sxfit), pxfit)
    if (returnf){
        return(result)
    }
    output_to_python <- toJSON(result, force = TRUE)
    write(output_to_python, filename)
}

for (tar in target_combo){
    print(paste(tar[1], tar[2], 'running'))
    flush.console()
    gam_proc2d(cattarget, dfraw, tar[1], tar[2], stg, fs, oversample, model_type)
}  
