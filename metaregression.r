library(mgcv)
library(dplyr)
library(tidyverse)
library(data.table)
library(arrow)

dfraw <- arrow::read_parquet(paste0(args[1],'.parquet'))
fit<-gam(y~s(x0)+s(site, bs='re')+auc, data=dfraw)
save(fit, paste0(args[1], file='_gam.RData')