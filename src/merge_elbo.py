import numpy as np
import sys
import os

import pandas as pd


def read_single_seed(dataset,seed,batch_size=1000):
    Variants_truth=pd.read_excel('Input/'+dataset+'/'+dataset+'.xlsx')['Variants'].values
    elbos=[]
    variants=[]
    num_batch=np.ceil(float(len(Variants_truth))/float(batch_size))
    # print(len(Variants_truth))
    # print(num_batch)
    # print(int(num_batch))
    for num in range(int(num_batch)):
        data=np.load('Input/'+dataset+'/'+'elbo_seed'+str(seed)+'_batch'+str(num)+'.npz')
        elbos.extend(data['elbos'])
        variants.extend(data['variants'])
    # print(len(Variants_truth))
    # print(len(variants))
    assert np.array_equal(Variants_truth,variants)
    return np.array(elbos)
dataset=sys.argv[1]
num_model=5
elbo=[]
for i in range(1,num_model+1):
    # path = os.path.join('inference', dataset, 'elbo'+'_seed'+str(i)+'.npy')
    # data = np.loadtxt(path)
    elbo.append(read_single_seed(dataset,seed=i))
elbo=np.asarray(elbo)
elbo=np.mean(elbo,axis=0)
np.save('Input/'+dataset+'/elbo.npy',elbo)

