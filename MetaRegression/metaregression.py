import numpy as np
import pandas as pd
from scipy import stats, optimize
import csv
import matplotlib.pyplot as plt
import pymare

class Curve:
    def __init__(self, inputX, inputY):
        self.x = inputX
        self.y = inputY

    def getMinX(self):
        return np.min(self.x)
    
    def getMaxX(self):
        return np.max(self.x)
        
    def normalizeCurve(self, minJ, maxJ):
        self.norX = (self.x-minJ)/(maxJ-minJ)*2-1
        self.minX = minJ
        self.maxX = maxJ

    def fitLegendre(self, deg):
        self.fit = np.polynomial.legendre.Legendre.fit(self.norX, self.y, deg, domain = [-1,1], full=True)
        
    def getFit(self):
        return self.fit[0]    

    def plotCurve(self, show=True):
        [fx, fy] = self.fit[0].linspace(domain=[self.minX, self.maxX])
        fig = plt.figure(1)
        plt.title("Fitted Legendgre Curve", fontsize='16')	
        plt.xlabel("X",fontsize='13')	
        plt.ylabel("log(oddRatio)",fontsize='13')	

        plt.plot(self.x, self.y, label='o')	
        plt.plot(fx, fy)	

        plt.legend(['raw', 'fitted'],loc='best')
        plt.grid()	
        
        if show:
            plt.show()
        return plt  

class Site:
    def __init__(self):
        self.curve = list()
    
    def addCurve(self, inputX, inputY):
        self.curve.append(Curve(inputX, inputY))

    def getCurve(self):
        return self.curve

    def getMetaCood(self):
        cood = list()
        for c in self.curve:
            cood.extend(c.getFit().coef)
        return np.array(cood)

    def addAUC(self,inputAUC,inputAUCvar):
        self.AUC = inputAUC
        self.AUCvar = inputAUCvar

    def getAUC(self):
        return self.AUC

    def getAUCvar(self):
        return self.AUCvar

numSite = 4
data = list()

# Read site data from cvs file
for i in range(numSite):
    siteData = Site()
    with open("D:\Postdoc\MetaRegression\Site"+str(i)+".csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_list = list(csv_reader)
        for i in range(0, len(csv_list)-2, 2):
            siteData.addCurve(np.array(csv_list[i], dtype=np.float32), np.array(csv_list[i+1], dtype=np.float32))
        siteData.addAUC(np.array(csv_list[len(csv_list)-2], dtype=np.float32)[0], np.array(csv_list[len(csv_list)-1], dtype=np.float32)[0])
    data.append(siteData)

# Normalize curve domain to [-1,1]
NumCurve = len(data[0].getCurve())
for j in range(NumCurve):
    minJ = min([siteData.getCurve()[j].getMinX() for siteData in data])
    maxJ = max([siteData.getCurve()[j].getMaxX() for siteData in data])
    for siteData in data:
        siteData.getCurve()[j].normalizeCurve(minJ, maxJ)

# Fit Legendre
deg = [10, 10, 10, 10]
for siteData in data:
    for j in range(len(siteData.getCurve())):
        siteData.getCurve()[j].fitLegendre(deg[j])

data[0].getCurve()[0].plotCurve(show=True)

# Meta Regression
y = np.array([siteData.getAUC() for siteData in data])
v = np.array([siteData.getAUCvar() for siteData in data])
X = data[0].getMetaCood()
for i in range(1, len(data)):
    X = np.concatenate((X, data[i].getMetaCood()), axis=0)

result = pymare.meta_regression(y, v, X, add_intercept=True, method='REML')
print(result.to_df())
