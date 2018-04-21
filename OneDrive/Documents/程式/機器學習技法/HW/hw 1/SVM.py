import numpy as np
import random
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def readToList02(filename):
    desList = []
    with open(filename) as file:
         for line in file:
             line = line.split()
             if line:
                 line = [float(i) for i in line]
                 desList.append(line)
    return desList

# Read in training data and test data
TrData = np.array(readToList02('features.train.txt'))
TeData = np.array(readToList02('features.test.txt')) 

# Prob 11.
## Set Data
for eachR in TrData:
    if eachR[0] != 1:
       eachR[0] = -1
TrY11 = TrData[:,0]
TrX11 = TrData[:,[1,2]]
C = [pow(10,-5), pow(10,-3), 0.1, 10, 1000]
clf11 = {} # create a dictionary to store different clf (due to diff C) in prob11 -> https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables-via-a-while-loop
for i in range(1, len(C)):
    clf11[i] = SVC(C = C[i], kernel = 'linear') # set model
    clf11[i].fit(TrX11, TrY11) # training
## get W
clf11WeiMeg = [] # store the magnitude of wieghts 
for i in range(1, len(clf11)):
    tmpW  = [np.linalg.norm(clf11[i].coef_)] # https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F
    clf11WeiMeg.append(tmpW)
## plot
plt.plot(np.log10(C), clf11WeiMeg)
plt.title("Prob 11")

# Prob 12
## Set Data
for eachR in TrData:
    if eachR[0] != 8:
       eachR[0] = -1
    else: eachR[0] = 1
TrY12 = TrData[:,0]
TrX12 = TrData[:,[1,2]]
clf12 = {} # create a dictionary to store different clf (due to diff C) in prob11 -> https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables-via-a-while-loop
for i in range(1, len(C)):
    clf12[i] = SVC(C = C[i], kernel = 'poly', coef0 = 1, gamma = 1, degree = 2) # set model
    clf12[i].fit(TrX12, TrY12) # training
Ein = []
## get Ein
for eachClf in range(1,len(clf12)):
    ein = [(TrY12[i] != clf12.predict(TrX12[i])) for i in range(1,len(TrData))] # a list recond every data predict result, 0 is correct
    Ein += [float(sum(ein))/len(TrData)]
## plot
plt.plot(np.log10(C), Ein)
plt.title("Prob 12")

# Prob 13
## get number of support 
nSupport = []
for i in range(1, len(clf12)):
    ns = sum(clf12[i].n_support_)
    nSupport += [ns]
## plot
plt.plot(np.log10(C), nSupport)
plt.title("Prob 13")

# Prob14
C2 = [pow(10,-3), pow(10,-2), 0.1, 1, 10]
clf14 = {}
## train
for i in range(1, len(C2)):
    clf14[i] = SVC(C = C[i], kernel = 'rbf', gamma = 80)
    clf14[i].fit(TrX11, TrY11)
anyFreeSVind = []
## choose a fSV (record its indice)
for i in range(1,len(clf14)):
    svind = clf14.support_ # indices of support vectors. (list)
    fsvind == C2[i] # fsvind is a choosen indice of a free support vector
    while fsvind == C2[i]: # is not a free SV
       fsvind = random.choice(svind)
    anyFreeSVind += fsvind 
## get distance
Dist = []
for i in range(1,len(clf14)):
    dist = clf14[i].decision_function(TrX11)[anyFreeSVind[i]]
    Dist += dist
## plot
plt.plot(np.log10(C), Dist)
plt.title("Prob 14")

# Prob15
for eachR in TeData:
    if eachR[0] != 1:
       eachR[0] = -1
TeY = TeData[:,0]
TeX = TeData[:,[1,2]]
Cfix = 0.1
Gamma = [1, 10, 100, 1000, 10000]
clf15 = {}
## Train
for i in range(1, len(Gamma)):
    clf15[i] = SVC(C = Cfix, kernel = 'rbf', gamma = Gamma[i])
    clf15.fit(TrX11, TrY11)
## Get Eout
Eout = []
for eachClf in range(1, len(clf15)):
    eout = [(clf15[i].predict(TeX[i]) != TeY[i]) for i in len(TeData)]
    Eout += [float(sum(eout))/len(TeData)]
## Plot
plt.plot(np.log10(Gamma), Eout)
plt.show()

# Prob16
valRepT = 100
for t in range(1, valRepT):
    randSmp = 
