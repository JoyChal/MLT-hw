import numpy as np
import random
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
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
for eachR in TeData:
    if eachR[0] != 1:
       eachR[0] = -1

# Prob 11.
## set Data
for eachR in TrData:
    if eachR[0] != 1:
       eachR[0] = -1
TrY11 = TrData[:,[0]]
TrX11 = TrData[:,[1,2]]
C = [pow(10,-5), pow(10,-3), 0.1, 10, 1000]
clf11 = [] # create a list to store different clf (due to diff C) in prob11 -> https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables-via-a-while-loop
for i in range(0, len(C)):
    clf = SVC(C = C[i], kernel = 'linear') # set model
    clf.fit(TrX11, TrY11) # training
    clf11.append(clf)
## get W
clf11WeiMeg = [] # store the magnitude of wieghts 
for i in range(0, len(clf11)):
    tmpW  = np.linalg.norm(clf11[i].coef_) # https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F
    clf11WeiMeg.append(tmpW)
## plot
plt.plot(np.log10(C), clf11WeiMeg)
plt.title("Prob 11")
plt.xlabel("log10(C)")
plt.ylabel("magnitude of W")
plt.show()

# Prob 12
## set Data
TrData = np.array(readToList02('features.train.txt'))
for eachR in TrData:
    if eachR[0] != 8:
       eachR[0] = -1
    else: eachR[0] = 1
TrY12 = TrData[:,[0]]
TrX12 = TrData[:,[1,2]]
clf12 = [] # create a dictionary to store different clf (due to diff C) in prob11 -> https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables-via-a-while-loop
for i in range(0, len(C)):
    clf = SVC(C = C[i], kernel = 'poly', coef0 = 1, gamma = 1, degree = 2) # set model
    clf.fit(TrX12, TrY12) # training
    clf12.append(clf)
Ein = []
## get Ein
for eachClf in range(0,len(clf12)):
	# original method
    '''ein = [(TrY12[i] != clf12.predict(TrX12[i])) for i in range(1,len(TrData))] # a list recond every data predict result, 0 is correct'''
    ein = 1-clf12[eachClf].score(TrX12, TrY12)
    Ein += [ein] # original : float(sum(ein))/len(TrData) 
## plot
plt.plot(np.log10(C), Ein)
plt.title("Prob 12")
plt.xlabel("log10(C)")
plt.ylabel("Ein")
plt.show()

# Prob 13
## get number of support 
nSupport = []
for i in range(0, len(clf12)):
    ns = sum(clf12[i].n_support_)
    nSupport += [ns]
## plot
plt.plot(np.log10(C), nSupport)
plt.title("Prob 13")
plt.xlabel("log10(C)")
plt.ylabel("number of SV")
plt.show()

# Prob14
C2 = [pow(10,-3), pow(10,-2), 0.1, 1, 10]
clf14 = []
## train
for i in range(0, len(C2)):
    clf = SVC(C = C[i], kernel = 'rbf', gamma = 80)
    clf.fit(TrX11, TrY11)
    clf14.append(clf)
anyFreeSVind = []
## choose a fSV (record its indice)
for i in range(0,len(clf14)):
    svind = clf14[i].support_ # indices of support vectors. (list)
    fsvind = C2[i] # fsvind is a choosen indice of a free support vector
    while fsvind == C2[i]: # is not a free SV
       fsvind = random.choice(svind)
    anyFreeSVind += [fsvind] 
## get distance
Dist = []
for i in range(0,len(clf14)):
    dist = abs(clf14[i].decision_function(TrX11)[anyFreeSVind[i]])
    Dist += [dist]
## plot
plt.plot(np.log10(C), Dist)
plt.title("Prob 14")
plt.xlabel("log10(C)")
plt.ylabel("distances of a free SV to hyperplane")
plt.show()

# Prob15
for eachR in TeData:
    if eachR[0] != 1:
       eachR[0] = -1
TeY = TeData[:,[0]]
TeX = TeData[:,[1,2]]
Cfix = 0.1
Gamma = [1, 10, 100, 1000, 10000]
clf15 = []
## train
for i in range(0, len(Gamma)):
    clf = SVC(C = Cfix, kernel = 'rbf', gamma = Gamma[i])
    clf.fit(TrX11, TrY11)
    clf15.append(clf)
## get Eout
Eout = []
for eachClf in range(0, len(clf15)):	
    # original method
    '''eout = [(clf15[i].predict(TeX[i]) != TeY[i]) for i in len(TeData)]'''
    eout = 1-clf15[eachClf].score(TeX, TeY)
    Eout += [eout] # original : float(sum(eout))/len(TeData)
## plot
plt.plot(np.log10(Gamma), Eout)
plt.title('Prob15')
plt.xlabel("log10(Gamma)")
plt.ylabel("Eout")
plt.show()

# Prob16
valRepT = 100
choosenT = [0]*5 # number of time that a given gamma is choosen  
for t in range(0, valRepT):
    valTrain, valTest = train_test_split(TeData,test_size=1000) 
    valTrX = valTrain[:,[1,2]]
    valTrY = valTrain[:,[0]]
    valTeX = valTest[:,[1,2]]
    valTeY = valTest[:,[0]]
    Eval = []
    for i in range(0, len(Gamma)):
        clf = SVC(C = Cfix, kernel = 'rbf', gamma = Gamma[i]).fit(valTrX, valTrY)
        eval = 1-clf.score(valTeX, valTeY) 
        '''print(eval)'''
        Eval += [eval]  
    mingamInd = Eval.index(min(Eval))
    '''print(mingamInd)'''
    choosenT[mingamInd] += 1 
    '''print(t)'''
## Plot
plt.bar(np.log10(Gamma), choosenT)
plt.title('Prob16')
plt.xlabel("log10(Gamma)")
plt.ylabel("selected time of a given gamma")
plt.show()