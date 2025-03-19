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
library(tidygam)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if arguments are provided
if (length(args) == 0) {
    stop("No arguments provided!")
}

arg1 <- args[1]

if (arg1=='single'){
    path <- '/home/hoyinchan/blue/Data/data2022/shapalltmp.parquet'
}

if (arg1=='interaction'){
    path <- '/home/hoyinchan/blue/Data/data2022/shapalltmp_interaction.parquet'
#    path <- '/home/hoyinchan/blue/Data/data2022/shapalltmp.parquet'
}

dfraw <- arrow::read_parquet(path)

#dfraw <- dfraw %>% drop_na()
dfraw <- dfraw %>% dplyr::select(-'__index_level_0__')
#dfraw <- dfraw %>% dplyr::filter(site_d != 'MCRI') %>% dplyr::filter(site_m != 'MCRI')
dfraw$site_d <- as.factor(dfraw$site_d)
dfraw$site_m <- as.factor(dfraw$site_m)
#dfraw <- dfraw %>% rename(Feature = feature) %>% rename(val = value)

column_names <- colnames(dfraw)
split_names <- strsplit(column_names, "_")
first_parts <- sapply(split_names, `[`, 1)
targets <- unique(first_parts)
targets <- setdiff(targets, "site")
targets <- setdiff(targets, "")
targets <- ifelse(targets == 'ORIGINAL', 'ORIGINAL_BMI', targets)


#cattarget <- list("PX:CH:J1940", "PX:09:96.72")
#cattarget <- names(df)[sapply(df, is.logical)]
cattargetdf <- arrow::read_parquet('/home/hoyinchan/code/AKI_CDM_PY/bool_columns.parquet')
cattarget <- cattargetdf[['index']]

##TEST
#targets <- targets[1:2]

fit_proc <- function(eqn, dfraw2, target, type, weight=FALSE){
    if (weight==FALSE){
        xfit <- bam(eqn, data=dfraw2, method='REML')  
    }else{
        xfit <- bam(eqn, data=dfraw2, method='REML', weight=rocw)  
    }
    print(target)
    print(type)
    print(summary(xfit))
    flush.console()
    sxfit<-summary(xfit)
    pxfit<-plot(xfit)
    pxfit2<-termplot(xfit, data=dfraw2, se = TRUE, plot = FALSE)
    for (i in 1:length(pxfit)){
        pxfit[[i]]$raw=NULL    
    }
    return(list(target, type, sxfit, pxfit, pxfit2))
}    

gam_proc <- function(outputname, cattarget, targets, dfraw, returnf=FALSE, weight=FALSE, noAUC=FALSE) {
    resultlist <- list()
    
    for (target in targets){
        print(target)
        flush.console()
#        dfraw2 <- dfraw %>% filter(Feature==target)            
        columns_to_select <- c(paste0(target, '_Names'), paste0(target, '_vals'), 'site_m', 'site_d')
        dfraw2 <- dfraw[,columns_to_select]
        colnames(dfraw2) <- c('Name', 'val', 'site_m', 'site_d')        
        if (target %in% cattarget){
            dfraw2$Name <- as.factor(dfraw2$Name)
            if (noAUC==FALSE){
                eqnl <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re") + roc2
                eqnq <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re") + roc2
                eqns <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re") + roc2
                eqnt <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re") + roc2 
                resultl<-fit_proc(eqnl, dfraw2, target, 'linear', weight=weight)
                resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic', weight=weight)
                results<-fit_proc(eqns, dfraw2, target, 'spline', weight=weight)
                resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction', weight=weight)            
                result<-list(resultl, resultq, results, resultt)
            }else{
                eqnl <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re")
                eqnq <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re")
                eqns <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re")
                eqnt <- val ~ Name + s(site_d,bs="re") + s(site_m,bs="re") 
                resultl<-fit_proc(eqnl, dfraw2, target, 'linear', weight=weight)
                resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic', weight=weight)
                results<-fit_proc(eqns, dfraw2, target, 'spline', weight=weight)
                resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction', weight=weight)            
                result<-list(resultl, resultq, results, resultt)
             }
        }else{
            if (noAUC==FALSE){
                eqnl <- val ~ poly(Name,1,raw=TRUE) + s(site_d,bs="re")  + s(site_m,bs="re") + roc2
                eqnq <- val ~ poly(Name,2,raw=TRUE) + s(site_d,bs="re")  + s(site_m,bs="re") + roc2
                eqns <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")  + s(site_m,bs="re") + roc2
                eqnt <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")  + s(site_m,bs="re") + roc2 + ti(Name,roc2,bs='cr')            
                resultl<-fit_proc(eqnl, dfraw2, target, 'linear')
                resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic')
                results<-fit_proc(eqns, dfraw2, target, 'spline')
                resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction')            
                result<-list(resultl, resultq, results, resultt)
            }else{
                eqnl <- val ~ poly(Name,1,raw=TRUE) + s(site_d,bs="re")  + s(site_m,bs="re")
                eqnq <- val ~ poly(Name,2,raw=TRUE) + s(site_d,bs="re")  + s(site_m,bs="re")
                eqns <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")  + s(site_m,bs="re")
                eqnt <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")  + s(site_m,bs="re")          
                resultl<-fit_proc(eqnl, dfraw2, target, 'linear', weight=weight)
                resultq<-fit_proc(eqnq, dfraw2, target, 'quadratic', weight=weight)
                results<-fit_proc(eqns, dfraw2, target, 'spline', weight=weight)
                resultt<-fit_proc(eqnt, dfraw2, target, 'spline_interaction', weight=weight)            
                result<-list(resultl, resultq, results, resultt)
            }
        }
        resultlist <- append(resultlist, list(result))
    }
    if (returnf){
        return(resultlist)
    }
    output_to_python <- toJSON(resultlist, force = TRUE, digit=30)
    write(output_to_python, paste0(outputname))
}



# outputname <- "gamalltmp_single_AUC_populationweight.json"
# gam_proc(outputname, cattarget, targets, dfraw, weight=TRUE, noAUC=FALSE)

start_time <- Sys.time()
print("Meta-Running univariate regression R")
outputname <- "/blue/yonghui.wu/hoyinchan/program_data/AKI_CDM_PY/MetaRegression/gamalltmp_single_noAUC.json"
if (arg1=='single'){
    gam_proc(outputname, cattarget, targets, dfraw, weight=FALSE, noAUC=TRUE)
}
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Meta-Running Finished univariate regression R in", runtime, " seconds")

cat("done")

  
if (arg1=='single'){
    quit()
}

# outputname <- "gamalltmp_single_weightAUC2.json"
# gam_proc(outputname, cattarget, targets, dfraw, weight=TRUE, noAUC=TRUE)

# outputname <- "gamalltmp_single_weightAUC2.json"
# gam_proc(outputname, cattarget, targets, dfraw, weight=TRUE, noAUC=TRUE)

# dfraw2 <- dfraw %>% filter(Feature=='LAB::2345-7(mg/dL)') 
# #eqnl <- val ~ poly(Name,1,raw=TRUE) + s(site_d,bs="re")   
# eqnq <- val ~ poly(Name,2,raw=TRUE) + s(site_d,bs="re")   
# #eqns <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")   
# #eqnt <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")    + ti(Name,roc2,bs='cr')            
# resultq<-fit_proc(eqnq, dfraw2, 'AGE', 'quadratic')
# #results<-fit_proc(eqns, dfraw2, 'AGE', 'spline')

# toJSON(resultq, force = TRUE, digit=30)

# summary(resultq)

# ## 2D

# dfraw2 <- dfraw %>% filter(Feature=='AGE')
# eqnl <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")   + roc2
# xfit <- bam(eqnl, data=dfraw2, method='REML') 
# plot(xfit)

# dfraw2 <- dfraw %>% filter(Feature=='AGE')
# eqnl <- val ~ s(Name,k=10,bs='cr') + s(site_d,bs="re")   + roc2 + s(site_d,roc2,bs="re")
# xfit <- bam(eqnl, data=dfraw2, method='REML') 
# plot(xfit)

# summary(xfit)

# Try 2 feature interaction

target_combo = combn(targets, 2, simplify = FALSE)
outputname <- "gamalltmp_double_interaction_quadratic.json"
reversed_target_combo <- lapply(target_combo, rev)
# Combine the original and reversed lists
combined_list <- c(target_combo, reversed_target_combo)

combined_list

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
#    if (TRUE){
        filename <- paste0('/blue/yonghui.wu/hoyinchan/program_data/AKI_CDM_PY/MetaRegression/gam2d_tmp/','gam2d_tmp','_',f1str,'_',f2str,'_',stg,'_',fs,'_',oversample,'_',model_type,'.json')
        if (file.exists(filename)){
            print(paste0('Exists: ','gam2d_tmp','_',f1str,'_',f2str,'_',stg,'_',fs,'_',oversample,'_',model_type,'.json'))
            return()
        }    
        dfraw <- arrow::read_parquet(path)
#        dfraw <- dfraw %>% drop_na()
        dfraw <- dfraw %>% dplyr::select(-'__index_level_0__')
        #dfraw <- dfraw %>% dplyr::filter(site_d != 'MCRI') %>% dplyr::filter(site_m != 'MCRI')
        dfraw$site_d <- as.factor(dfraw$site_d)
        dfraw$site_m <- as.factor(dfraw$site_m)
#        dfraw <- dfraw %>% rename(Feature = feature) %>% rename(val = value)
        # if (file.exists(filename)){
        #     return()
        # }
    }
    
    eqn_cc <- val ~ s(Name.x,k=10,bs='cr') + s(Name.y,k=10,bs='cr') + s(site_d,bs="re") + s(site_m,bs="re") + ti(Name.x,Name.y,k=10,bs='cr')
    eqn_cd <- val ~ s(Name.x,k=10,bs='cr') + s(Name.x,by=Name.y,k=10,bs='cr') + Name.y + s(site_d,bs="re") + s(site_m,bs="re")

    eqn_cs <- val ~ s(Name.x,k=10,bs='cr') + ti(Name.x,Name.y,k=10,bs='cr') + s(site_d,bs="re") + s(site_m,bs="re")
    
    if (f1 %in% cattarget & !f2 %in% cattarget){
        tmp = f1
        f1 = f2
        f2 = tmp
    }

    
    columns_to_select <- c(paste0(f1, '_Names'), paste0(f1, '_vals'), paste0(f2, '_Names'), paste0(f2, '_vals'), 'site_m', 'site_d')
    dfraw23 <- dfraw[,columns_to_select]
    colnames(dfraw23) <- c('Name.x', 'val.x','Name.y', 'val.y', 'site_m', 'site_d')
    dfraw23 <- dfraw23 %>% mutate(val=val.x+val.y)
        
    
    
#    dfraw21 <- dfraw %>% filter(Feature==f1)
#    dfraw22 <- dfraw %>% filter(Feature==f2)
    
#    dfraw23 <- dfraw21 %>% inner_join(dfraw22, by=c('ID','site_d', 'site_m'))
#    dfraw23 <- dfraw23 %>% mutate(val=val.x+val.y)
#    dfraw23 <- dfraw23 %>% mutate(val=val.x)

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
    output_to_python <- toJSON(result, force = TRUE, digit=30)
    write(output_to_python, filename)    
}

read_config <- function(file_path) {
  # Read the lines from the file
  lines <- readLines(file_path)
  
  # Initialize an empty list to store the configuration
  config <- list()
  
  # Iterate over each line
  for (line in lines) {
    # Split the line into key and value at the first '='
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    
    # Trim any leading or trailing whitespace from key and value
    key <- trimws(parts[1])
    value <- trimws(parts[2])
    
    # Convert logical values
    if (value == "True") {
      value <- TRUE
    } else if (value == "False") {
      value <- FALSE
    }
    
    # Convert numerical values
    if (grepl("^-?[0-9.]+$", value)) {
      value <- as.numeric(value)
    }
    
    # Add the key-value pair to the list
    config[[key]] <- value
  }
  
  return(config)
}

file_path <- "/home/hoyinchan/code/AKI_CDM_PY/configs_files/publish_config/configs_KUMC.txt"
config <- read_config(file_path)
#print(config)

# Load libraries
library(foreach)
library(doParallel)

# Register parallel backends
numCores <- 10  # Or set a specific number
registerDoParallel(cores=numCores)

start_time <- Sys.time()
print("Meta-Running multivariate regression R")
# Parallel foreach loop
foreach(targetc = iter(target_combo)) %dopar% {
#for (tar in target_combo){
    tryCatch({
        print(paste(targetc[1], targetc[2], 'running'))
        flush.console()
        gam_proc2d(cattarget, dfraw, targetc[1], targetc[2], config$stg, config$fs, config$oversample, config$model_type)
    },
    error = function(cond) {cat(cond)}
    )
}  
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Meta-Running Finished multivariate regression R in", runtime, " seconds")

print('done')

#gam_proc2d(cattarget, dfraw, targets[1], targets[2], config$stg, config$fs, config$oversample, config$model_type)

