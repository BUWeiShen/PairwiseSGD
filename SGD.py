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


def draw(ave_norm_diff):
    '''
    Plot Parameter Distance
    '''
    
    plt.plot(range(len(ave_norm_diff)),ave_norm_diff,'--',label='average')
    plt.xlabel('iterations')
    plt.ylabel('average parameter distance')
    plt.legend()
    
    return





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

        
        
    
    #stability measure
    # Read data 
    dataset = 'fourclass'
    hf = h5py.File('/Users/weishen/Desktop/%s.h5' % (dataset), 'r')
    FEATURES_all = hf['FEATURES'][:]
    LABELS_all = hf['LABELS'][:]
    N,d = FEATURES_all.shape

    # create S and S' which are different only in one position
    delet = np.random.randint(N)
    FEATURES_1 = list(FEATURES_all)
    LABELS_1 = list(LABELS_all)
    Delement_F = FEATURES_1[delet]
    Delement_L = LABELS_1[delet]
    del FEATURES_1[delet]
    del LABELS_1[delet]

    repla = np.random.randint(N-1)
    FEATURES_2 = list(FEATURES_1)
    LABELS_2 = list(LABELS_1)
    FEATURES_2[repla] = Delement_F
    LABELS_2[repla] =  Delement_L
    
    N = N-1
    hf.close()

    # Define hyper-parameters
    epochs = 1
    K = 10 
    T = N-1+(epochs-1)*N
    # Define parameters
    #mu = 1
    #R = np.sqrt(2/mu)
    
    mu=0
    R = 100


    

    sum_norm_diff = np.zeros(T)
    
    # out-loop k for averaged measure
    for k in range(1,K+1):
        w_1 = np.zeros(d)
        w_2 = np.zeros(d)
        number_list = list(range(1,N+1))
        random.shuffle(number_list)  
        norm_diff = []
        
    
    # Run SGD with random permutation among epoch
        for j in range(1,epochs+1):
            for i in range(max(2,N*(j-1)+1),N*j+1):
                #for t in range(max(i-20,1),i):
                for t in range(i):
                    eta = 0.1 / (i-1)
                    w_1 -= eta / (i-1) * gAUC(w_1,FEATURES_1[number_list[i%N-1]-1],LABELS_1[number_list[i%N-1]-1],FEATURES_1[number_list[t%N-1]-1],LABELS_1[number_list[t%N-1]-1])
                    w_2 -= eta / (i-1) * gAUC(w_2,FEATURES_2[number_list[i%N-1]-1],LABELS_2[number_list[i%N-1]-1],FEATURES_2[number_list[t%N-1]-1],LABELS_2[number_list[t%N-1]-1])
            
                norm_1 = np.linalg.norm(w_1)
                if norm_1 > R:
                    w_1 = w_1 / norm_1 * R
                
                norm_2 = np.linalg.norm(w_2)
                if norm_2 > R:
                    w_2 = w_2 / norm_2 * R
            
                norm_diff.append(np.linalg.norm(w_1-w_2))
                        

                #if i % 100 == 0:
                    #diff = np.linalg.norm(w_1-w_2)
                    #print(diff)
                    #er = pairer(AUC, w_1, FEATURES_all, LABELS_all)
                    #print(norm_diff)
                
                    #print('iteration: %d empirical risk: %f' %(i,er))
            # permutation again after every epoch        
            random.shuffle(number_list)  
        sum_norm_diff +=  np.array(norm_diff)
    
    ave_norm_diff = sum_norm_diff / K
    draw(ave_norm_diff)
