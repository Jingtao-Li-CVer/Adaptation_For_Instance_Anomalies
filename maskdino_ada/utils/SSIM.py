# -*- coding: utf-8 -*-
"""
Created on  2020/10/28

@author: Xu Buyun


-------------------------------------------------------------------------------
"""
import os
from scipy.io import loadmat,savemat

import numpy as np
import math

from skimage.metrics import structural_similarity as compare_ssim
from scipy.spatial.distance import pdist,squareform
#%%

def ssim_similarity_matrix(hyperIm,w=3):
    sizeBand = hyperIm.shape[-1]
    List=[]
    for i in range(0,sizeBand):
        X=hyperIm[:,:,i]
        for j in range(i+1,sizeBand):
            Y=hyperIm[:,:,j]
            List.append(compare_ssim(X,Y,gaussian_weight=True,win_size=w, data_range=1.0))
    Y=np.array(List)
    Mat_ssim=squareform(List)    
    return Mat_ssim
    
def SR(S,K):
    '''
    Algorithm: The Similarity-based Ranking (SR) Algorithm 
    Input: S is the similarity matrix
           K is the desired number of clusters
    Output:M is the indexes of cluster centers
    '''
    # sc
    L = S.shape[0]
    Y = S[np.triu_indices(L,k=1)]
    Y = sorted(Y,reverse=True)
    sc = np.mean(Y[math.floor( len(Y) * 0.05 )-1: math.floor(len(Y)*0.1)-1]) 

        
    # Step1: Average Similarity
    alpha=[]
    for i in range(0,L):
        temp=S[i,:]
        if len(temp[temp>sc])==0:
            alpha.append(0)
        else:
            alpha.append(np.mean(temp[temp>sc]))
    alpha=np.array(alpha)
    I=np.argsort(-alpha)
    
    
    # Step2: relative distance
    varphi = np.zeros(L)
    varphi[I[0]] = 1
    A = [0]*L
    for i in range(1,L):
        for j in range(i):
            if varphi[I[i]] < S[I[i], I[j]]:
                varphi[I[i]] = S[I[i], I[j]]
                A[I[i]] = I[j]
    varphi[I[0]] = min(varphi)
    A[I[0]] = I[0]
    theta = np.sqrt(1-varphi**2)
    
    # Step3: The Score of Each Band    
    alpha = (alpha - min(alpha)) / (max(alpha) - min(alpha))
    theta = (theta - min(theta)) / (max(theta) - min(theta))
    eta = alpha * theta
    
    # Step4: The Indexes of Cluster Centers
    G=np.argsort(-eta)
    M=np.sort(G[0:K])
    M=list(M)
               
    return M

#%%            
if __name__=='__main__':
    outpath = '../results/BandSelection_forHSI'
    if os.path.exists(outpath)==False:
            os.mkdir(outpath)
    ##HSInames=['indian_pines_185','salinas_corrected','KSC','Botswana']
    HSInames=['indian_pines_185']
    for Hi in range(len(HSInames)):
        HSIname = HSInames[Hi]
        hyperIm=loadmat('../data/'+HSIname+'.mat')[HSIname]
        #%%       
        ## SR_SSIM_w3
        # save path   
        outpathname = 'SR_SSIM_w3'
        outpath=('../results/BandSelection_forHSI/'+outpathname)
        if os.path.exists(outpath)==False:
            os.mkdir(outpath)
        # similarity matrix
        D = ssim_similarity_matrix(hyperIm,w=3)       
        # perform SR for band selection
        sltBands=[]
        for K in range(1,51,1):            
            M = SR(D, K)
            sltBands.insert(K-1,M) 
        sltBands=np.array(sltBands,dtype=object)
        varname='_'.join(['sltBands',outpathname,HSIname])
        savemat(outpath+'/'+varname+'.mat',{varname:sltBands})    
        #%%
        ## SR_SSIM
        # save path   
        outpathname = 'SR_SSIM'
        outpath=('../results/BandSelection_forHSI/'+outpathname)
        if os.path.exists(outpath)==False:
            os.mkdir(outpath)
        # similarity matrix
        D = ssim_similarity_matrix(hyperIm,w=11)       
        # perform SR for band selection
        sltBands=[]
        for K in range(1,51,1):            
            M = SR(D, K)
            sltBands.insert(K-1,M) 
        sltBands=np.array(sltBands,dtype=object)
        varname='_'.join(['sltBands',outpathname,HSIname])
        savemat(outpath+'/'+varname+'.mat',{varname:sltBands})  