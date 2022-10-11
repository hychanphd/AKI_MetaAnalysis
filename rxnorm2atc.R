library(rxnorm)
library(dplyr)

rxnorm_data <- read.delim("/home/hchan2/AKI/AKI_Python/rxnormtmp.csv", sep=",", stringsAsFactors=F, header = FALSE)
rxnorm_data <- rxnorm_data %>% mutate(V2=sapply(V1, rxnorm::get_atc))

write.csv(x=rxnorm_data, file='/home/hchan2/AKI/AKI_Python/rxnorm2atcdict.csv', row.names = FALSE)