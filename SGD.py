"""
STABILITY AND GENERALIZATION OF STOCHASTIC GRADIENT DESCENT FOR PAIRWISE LEARNING

Author:
"""

import numpy as np
import h5py
import random
from matplotlib import pyplot as plt

def pairsgd(gloss,w,eta,x,y):

    '''
    SGD for pairwise loss

    Input:
        gloss - gradient of loss function
        x - feature
        y - label
        eta - step size
        w - scoring function
        R - feasible radius

    Output:
        w - scoring function
    '''

    n = len(y)

    if n == 1:
        return w

    for i in range(n-1):
        w = w - eta / (n-1) * gloss(w,x[-1],y[-1],x[i],y[i])

    norm = np.linalg.norm(w)
    if norm > R:
        w = w / norm * R

    return w

def AUC(w,x1,y1,x2,y2):

    '''
    AUC loss

    Input:
        w - scoring function
        x1 -
        y1 -
        x2 -
        y2 -

    Output:
        auc - AUC loss
        gauc - AUC loss gradient
    '''
    prod = np.inner(x1 - x2,w)

    auc = (1 - prod)**2 * (y1+1)//2 * (1-y2)//2 + mu/2*np.linalg.norm(w)

    return auc

def gAUC(w,x1,y1,x2,y2):

    prod = np.inner(x1 - x2, w)

    gauc = 2*(prod-1) * (y1+1)//2 * (1-y2)//2 * (x1-x2) + mu*w

    return gauc

def pairer(loss,w,x,y):

    '''
    Pairwise empirical risk

    Input:
        loss -
        w -
        x -
        y -

    Ouput:
        er - empirical risk
    '''
    er = 0.0
    n = len(y)
    for i in range(n):
        for j in range(i):
            er += loss(w,x[i],y[i],x[j],y[j])

    er = er/(n*(n-2))

    return er

# wei 
def Birank(w,x1,y1,x2,y2):

    '''
    AUC loss
    Input:
        w - scoring function
        x1 -
        y1 -
        x2 -
        y2 -
    Output:
        auc - AUC loss
        gauc - AUC loss gradient
    '''
    prod = np.inner(x1 - x2,w)

    birank = (1 - (y1-y2)*prod)**2  + mu/2*np.linalg.norm(w)

    return birank

def gBirank(w,x1,y1,x2,y2):

    prod = np.inner(x1 - x2, w)

    gBirank = 2*((y1-y2)*prod-1) * (y1-y2) * (x1-x2) + mu*w

    return gBirank

if __name__ == '__main__':

    # Read data
    dataset = 'fourclass'
    hf = h5py.File('/Users/weishen/Desktop/%s.h5' % (dataset), 'r')
    FEATURES = hf['FEATURES'][:]
    LABELS = hf['LABELS'][:]
    hf.close()

    # Define hyper-parameters
    epochs = 2
    N,d = FEATURES.shape

    # Define parameters
    mu = 1
    R = np.sqrt(2/mu)
    eta = .01

    # Run SGD with fixed permutation among epoch
    aucw = np.zeros(d)
    for i in range(2,epochs*N):
        for t in range(i):
            aucw -= eta / (i-1) * gAUC(aucw,FEATURES[i%N],LABELS[i%N],FEATURES[t%N],LABELS[t%N])
            norm = np.linalg.norm(aucw)
            if norm > R:
                aucw = aucw / norm * R

        if i % 100 == 0:
            aucer = pairer(AUC, aucw, FEATURES, LABELS)
            print('iteration: %d empirical risk: %f' %(i,aucer))
          
    
    # Run SGD for BIRANK with random selection among all epoches (varying step sizes)
    barw = np.zeros(d)
    randslt = [np.random.randint(N)]
    for i in range(2,epochs*N):
        ind = np.random.randint(N)
        eta = 0.1 / (i-1)
        for t in randslt:
            barw -= eta / (i-1) * gBirank(barw,FEATURES[ind],LABELS[ind],FEATURES[t],LABELS[t])
            norm = np.linalg.norm(barw)
        if norm > R:
            barw = barw / norm * R
            
        #barw = ((i-1)*barw + w ) / i
        
        if i % 100 == 0:
            barer = pairer(Birank, barw, FEATURES, LABELS)
            print('iteration: %d empirical risk: %f' %(i,barer))
            
        randslt.append(ind)
        
        
        if __name__ == '__main__':

 

    

    

    # Run SGD with different random permutations among epochs
    # first permutation
    number_list = list(range(1,N+1))
    random.shuffle(number_list)
    # initial value
    w = np.ones(d)
    # first loop: epochs
    for j in range(1,epochs+1):
        # second loop: new example
        for i in range(max(2,N*(j-1)+1),N*j+1):
            # third loop: preview examples
            for t in range(i):
                eta = .1 / (i-1)
                w -= eta / (i-1) * gAUC(w,FEATURES[number_list[i%N-1]-1],LABELS[number_list[i%N-1]-1],FEATURES[number_list[t%N-1]-1],LABELS[number_list[t%N-1]-1])

            norm = np.linalg.norm(w)
            if norm > R:
                w = w / norm * R

            if i % 500 == 0:
                er = pairer(AUC, w, FEATURES, LABELS)
                print(w)
                print('iteration: %d empirical risk: %f' %(i,er))
        # new permutation again after every epoch        
        random.shuffle(number_list) 
