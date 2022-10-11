import numpy as np
import pandas as pd
import csv

numSite = 4

for i in range(numSite):
    CreatSerPI_x = np.sort(np.random.sample(30,)*2-1)
    CreatSerPI_y = np.sort(np.random.sample(30,)*5-1)
    Vancomycin_x = np.sort(np.random.sample(7,)*6+1)
    Vancomycin_y = np.sort(np.random.sample(7,)*0.25+0.5)
    BMI_x = np.sort(np.random.sample(100,)*50+10)
    BMI_y = np.sort(np.random.sample(100,)-0.25)
    BPSYSTOLIC_x = np.array([0, 1])
    BPSYSTOLIC_y = np.random.sample(2,)*0.35-0.1
    AUC = np.array([np.random.sample()*0.5+0.5])
    AUCvar = np.random.sample()*0.1*AUC
    data = [CreatSerPI_x, CreatSerPI_y, Vancomycin_x, Vancomycin_y, BMI_x, BMI_y, BPSYSTOLIC_x, BPSYSTOLIC_y, AUC, AUCvar]
    with open("D:\Postdoc\MetaRegression\Site"+str(i)+".csv", "w", newline='') as csv_file:
         writer = csv.writer(csv_file, delimiter=',')
         for line in data:
            writer.writerow(line)