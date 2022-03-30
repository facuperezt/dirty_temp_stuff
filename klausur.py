import numpy as np
def gridsearch(Area):
    minVal = float('inf')
    values = np.arange(Area[0],Area[1],0.1)
    for i in range(len(values)):
        t1 = values[i]
        for j in range(len(values)):
            t2 = values[j]
            if t1 +t1 < 10:
                tmp = getLogP(D,np.asarray((t1,t2)))
                if tmp < minVal:
                    minVal = tmp
    return minVal


def getLogP(D,THETA):
    D = D[0:k] # assuming k is the number of datapoint we want to use
    logP = [(D[i] -THETA[i,0])**2 + ( D[i]-THETA[i,1])**2 - np.log(np.pi*4) for i in range(D.shape[0])]
    logp = np.asarray(logP)
    return logP

N =5
k = 3


THETA = np.ones((k,2))
print(THETA)
D = np.ones(N)

getLogP(D,THETA)
#gridsearch((-10,10))