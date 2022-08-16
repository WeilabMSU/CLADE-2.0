import numpy as np
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
import warnings
import pickle
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import entropy,linregress

def linear_prob(a,threshold=0.0):
    prob=a/(np.sum(a))
    threshold=min(threshold,np.min(prob))
    prob[prob<threshold]=0.0
    prob=prob/(np.sum(prob))
    return prob
def softmax_prob(a,alpha):
    prob=np.exp(a*alpha)/(np.sum(np.exp(a*alpha)))
    return prob
def zero_sampling(features_zeroshot,Index,alpha=1.0,threshold=0.0):
    if len(Index)>1:
        cluster_mean=[[] for _ in range(features_zeroshot.shape[1])]
        for i in range(len(Index)):
            for k in range(features_zeroshot.shape[1]):
                cluster_mean[k].append((features_zeroshot[np.array(Index[i]),k].mean()-features_zeroshot[:,k].min())
                                                /(features_zeroshot[:,k].max()-features_zeroshot[:,k].min()))
        cluster_mean=np.array(cluster_mean)


        slope = []
        for i in range(cluster_mean.shape[0]):
            ss, _, _, _, _ = linregress(np.linspace(0, 1, len(cluster_mean[i,:])), np.sort(cluster_mean[i,:]))
            # print(slope)
            slope.append(ss)
        slope = np.array(slope)

        pp=linear_prob(slope)
        XX=[[] for _ in range(cluster_mean.shape[0])]
        for i in range(cluster_mean.shape[0]):
            XX[i]=softmax_prob(cluster_mean[i],alpha)
            # XX[i]=linear_prob(cluster_mean[i])
        XX=np.array(XX)
        prob=np.sum(XX*np.repeat(pp[:,np.newaxis],XX.shape[1],axis=1),axis=0)
        prob = linear_prob(prob)
    else:
        prob=np.ones(1)
    return prob

def shuffle_index(Index):
    for i in range(len(Index)):
        np.random.shuffle(Index[i])

    return Index
def next_sample(Prob,Fitness,AACombo,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index):
    for cluster_id in range(len(Index)):
        if len(Index[cluster_id]) == 0:
            Prob[cluster_id] = 0
    Prob = linear_prob(Prob)
    # Prob = Prob/np.sum(Prob)

    cluster_id = np.random.choice(np.arange(0, len(Index)), p=Prob)

    Fit[cluster_id].append(Fitness[Index[cluster_id][0]])
    SEQ[cluster_id].append(AACombo[Index[cluster_id][0]])
    Fit_list.append(Fitness[Index[cluster_id][0]])
    SEQ_list.append(AACombo[Index[cluster_id][0]])
    SEQ_index[cluster_id].append(Index[cluster_id][0])
    Index[cluster_id] = np.delete(Index[cluster_id], [0])

    return cluster_id,Prob,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index
def run_Clustering( features, n_clusters, subclustering_index=np.zeros([0])):
    if len(subclustering_index) > 0:
        features_sub = features[subclustering_index, :]
    else:
        features_sub=features

    kmeans = KMeans(n_clusters=n_clusters).fit(features_sub)
    cluster_labels = kmeans.labels_

    Length = []
    Index = []

    if len(subclustering_index) > 0:
        for i in range(cluster_labels.max() + 1):
            index = subclustering_index[np.where(cluster_labels == i)[0]]
            l = len(index)
            Index.append(index)
            Length.append(l)
    else:
        for i in range(cluster_labels.max() + 1):
            index = np.where(cluster_labels == i)[0]
            l = len(index)
            Index.append(index)
            Length.append(l)

    return Index
def ncluster_next_hierarchy(n_clusters_subclustering,Prob,Index):
    num_new_cluster = np.floor(Prob * n_clusters_subclustering).astype(int)
    for cluster_id in range(len(Prob)):
        num_new_cluster[cluster_id] = max(min(num_new_cluster[cluster_id], len(Index[cluster_id]) - 1), 0)
    meanfit_argsort = np.argsort(Prob)[::-1]
    meanfit_idx = 0
    while np.sum(num_new_cluster) < n_clusters_subclustering:
        cluster_id = meanfit_argsort[meanfit_idx]
        tmp = num_new_cluster[cluster_id] + n_clusters_subclustering - num_new_cluster.sum()
        num_new_cluster[cluster_id] = max(min(tmp, len(Index[cluster_id]) - 1), 0)
        meanfit_idx += 1
    assert np.sum(num_new_cluster) == n_clusters_subclustering
    return num_new_cluster
def split_subcluster(features, n_clusters, Index, Fitness, AACombo, SEQ_index, cluster_id):
    subclustering_index = []
    subclustering_index.extend(Index[cluster_id])
    subclustering_index.extend(SEQ_index[cluster_id])
    subclustering_index=np.asarray(subclustering_index)
    # subclustering_index = Index[cluster_id]
    # print(n_clusters)
    # print(subclustering_index)
    Index2 = run_Clustering( features, n_clusters, subclustering_index)
    Fit_sub = [[] for _ in range(n_clusters)]
    SEQ_sub = [[] for _ in range(n_clusters)]
    SEQ_index_sub = [[] for _ in range(n_clusters)]

    Index_sub=copy.deepcopy(Index2)


    for k in SEQ_index[cluster_id]:

        for i in range(len(Index2)):
            if k in Index2[i]:
                Fit_sub[i].append(Fitness[k])
                SEQ_sub[i].append(AACombo[k])
                SEQ_index_sub[i].append(k)
                Index_sub[i]=np.delete(Index_sub[i],np.where(Index_sub[i]==k))

    return Fit_sub, SEQ_sub, SEQ_index_sub, Index_sub
def cluster_mean_fit(Fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Mean_Fit = np.asarray([np.asarray(Fit[i]).mean() for i in range(len(Fit))])
    Mean_Fit[np.where(np.isnan(Mean_Fit))[0]] = 0
    return Mean_Fit

def sample_min_cluster(min_num_cluster, Fitness, AACombo, Index, Fit, SEQ, SEQ_index):

    num_add = 0
    for i in range(len(Index)):
        if len(Fit[i]) < min_num_cluster:
            for k in range(min_num_cluster - len(Fit[i])):
                Fit[i].append(Fitness[Index[i][k]])
                SEQ[i].append(AACombo[Index[i][k]])
                SEQ_index[i].append(Index[i][k])
                num_add += 1

        Index[i] = np.delete(Index[i], list(range(0, min_num_cluster - len(Fit[i]))))
    return Index, Fit, SEQ, SEQ_index, num_add

def cluster_sample(args,save_dir,features,features_zeroshot,AACombo, Fitness,ComboToIndex):
    K_increments = args.K_increments
    for i in range(len(K_increments)):
        K_increments[i]=int(K_increments[i])
    K_zeroshot = args.K_zeroshot
    N_hierarchy=len(K_increments)
    # encoding = args.encoding
    # dataset=args.dataset
    num_first_round=int(args.num_first_round)
    batch_size=int(args.batch_size)
    hierarchy_batch=int(args.hierarchy_batch)
    num_batch=int(args.num_batch)
    num_training_data = batch_size*num_batch
    alpha=args.softmax_beta
    # threshold=args.threshold


    # new hierarchy needs to be generated when number of samples is included in the array
    # hierarchy_first_round=True
    # if hierarchy_first_round:
    new_hierarchy = (-1+np.arange(1,N_hierarchy))*hierarchy_batch+num_first_round
    # else:
    #     new_hierarchy = np.arange(0,N_hierarchy)*hierarchy_batch+num_first_round

    # hierarchy = 0
    if K_zeroshot>0:
        n_clusters=K_zeroshot

        total_clusters = n_clusters
        ## run initial clustering use zeroshot features
        Index = run_Clustering(features_zeroshot, n_clusters)
        Index = shuffle_index(Index)

        # print(zeroshot_cluster_mean.shape)
        Prob = zero_sampling(features_zeroshot,Index,alpha=alpha)
    else:
        n_clusters=1
        total_clusters=n_clusters
        Index = [np.arange(features_zeroshot.shape[0])]
        Index = shuffle_index(Index)
        Prob = np.array([1.0])
    parents = [-1 * np.ones(len(Index))]
    hierarchy = 0
    hierarchy_first_round=True
    if hierarchy_first_round:
        n_clusters_subclustering=K_increments[hierarchy]
        num_new_cluster = \
            ncluster_next_hierarchy(n_clusters_subclustering, Prob, Index)

        Index_tmp=[]
        for cluster_id in range(len(Index)):
            if int(num_new_cluster[cluster_id])>0:
                Index_tmp.extend(run_Clustering(features,num_new_cluster[cluster_id]+1, Index[cluster_id]))
                parents[hierarchy][cluster_id]=cluster_id
                parents[hierarchy]=np.append(parents[hierarchy],cluster_id*np.ones(num_new_cluster[cluster_id]))
            else:
                Index_tmp.append(Index[cluster_id])
        Index=Index_tmp
        total_clusters=total_clusters+n_clusters_subclustering
        if K_zeroshot > 0:
            Prob = zero_sampling(features_zeroshot,Index,alpha=alpha)
        else:
            Prob = np.ones([total_clusters]) / total_clusters

    # store selected samples with sequential order
    Fit_list = []
    SEQ_list = []
    Cluster_list=[]
    #  store selected samples according to the cluster they belong to
    Fit = [[] for _ in range(len(Index))]
    SEQ = [[] for _ in range(len(Index))]
    SEQ_index = [[] for _ in range(len(Index))]

    # initial sampling probability is driven by zeroshot clustering
    num = 0
    print(Prob)
    while num < num_first_round:
        cluster_id,Prob,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index\
            =next_sample(Prob,Fitness,AACombo,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index)
        num += 1
    Mean_Fit = cluster_mean_fit(Fit)

    ## use tree structure to store hirearchy
    sample_length = np.zeros([len(Index)])
    for cluster_id in range(len(Index)):
        sample_length[cluster_id] = len(SEQ[cluster_id])
    tree = [[]]

    tree[hierarchy] = {'parents': copy.deepcopy(parents[hierarchy]), 'mean': copy.deepcopy(np.asarray(Mean_Fit)),
                         'num_samples': copy.deepcopy(np.asarray(sample_length)), 'Index': copy.deepcopy(Index),
                         'SEQ_index': copy.deepcopy(SEQ_index),
                          'initial_Prob':Prob}
    Prob = linear_prob(Mean_Fit)
    Index = shuffle_index(Index)


    while num < num_training_data:
        # generate new hierarchy
        if num in new_hierarchy:
            print(num)
            # if hierarchy_first_round:
            hierarchy+=1
            n_clusters_subclustering=K_increments[hierarchy]
            parents.append(-1 * np.ones([total_clusters]))
            tree.append({})
            if n_clusters_subclustering>0:
                # else:
                #     hierarchy+=1
                #     n_clusters_subclustering=K_increments[hierarchy-1]
                num_new_cluster=\
                    ncluster_next_hierarchy(n_clusters_subclustering, Prob, Index)

                for cluster_id in range(total_clusters):
                    if num_new_cluster[cluster_id] >= 1:
                        Fit_sub, SEQ_sub, SEQ_index_sub, Index_sub = \
                            split_subcluster( features, num_new_cluster[cluster_id] + 1, Index, Fitness,
                                             AACombo, SEQ_index, cluster_id)
                        # print(str(cluster_id)+' '+ str(len(Fit_sub)) +' ' +str(int(num_new_cluster[cluster_id])))
                        assert len(Fit_sub)==num_new_cluster[cluster_id]+1
                        Fit[cluster_id] = Fit_sub[0]
                        SEQ[cluster_id] = SEQ_sub[0]
                        SEQ_index[cluster_id] = SEQ_index_sub[0]
                        # print('hhh')
                        # print(len(Index))
                        Index[cluster_id] = Index_sub[0]
                        # print(len(Index))
                        # print(len(Index_sub))
                        for k in range(1, len(Fit_sub)):
                            Fit.append(Fit_sub[k])
                            SEQ.append(SEQ_sub[k])
                            SEQ_index.append(SEQ_index_sub[k])
                            Index.append(Index_sub[k])
                        parents[hierarchy][cluster_id] = cluster_id
                        parents[hierarchy] = np.append(parents[hierarchy], cluster_id * np.ones([len(Fit_sub) - 1]))

                total_clusters = n_clusters_subclustering + total_clusters
                assert np.sum(num_new_cluster)==n_clusters_subclustering
                ## update tree structure and randomly shuffle Index;
                Mean_Fit = cluster_mean_fit(Fit)
                Prob = linear_prob(Mean_Fit)
                Index = shuffle_index(Index)

                sample_length = np.zeros([len(Index)])
                for cluster_id in range(len(Index)):
                    sample_length[cluster_id] = len(SEQ[cluster_id])

                tree[hierarchy] = {'parents': copy.deepcopy(parents[hierarchy]),
                                   'mean': copy.deepcopy(np.asarray(Mean_Fit)),
                                     'num_samples': copy.deepcopy(np.asarray(sample_length)),
                                   'Index': copy.deepcopy(Index),
                                     'SEQ_index': copy.deepcopy(SEQ_index)}
        # update sampling probabilities and update sampling priority
        if np.mod(num, batch_size) == 0:
            Mean_Fit = cluster_mean_fit(Fit)
            Prob = linear_prob(Mean_Fit)
            Index = shuffle_index(Index)


        # select next sample
        cluster_id,Prob,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index\
            =next_sample(Prob,Fitness,AACombo,Fit,SEQ,Fit_list,SEQ_list,SEQ_index,Index)
        sample_length[cluster_id] = len(SEQ[cluster_id])

        tree[hierarchy]['num_samples'] = copy.deepcopy(np.asarray(sample_length))
        tree[hierarchy]['mean'] = copy.deepcopy(np.asarray(Mean_Fit))
        tree[hierarchy]['Index'] = copy.deepcopy(Index)
        tree[hierarchy]['SEQ_index'] = copy.deepcopy(SEQ_index)
        num += 1


    Fit_list = np.asarray(Fit_list)
    SEQ_list = np.asarray(SEQ_list)

    for seq in SEQ_list:
        for cluster_id in range(len(SEQ_index)):
            if ComboToIndex.get(seq) in SEQ_index[cluster_id]:
                Cluster_list.append(cluster_id)

    Cluster_list=np.asarray(Cluster_list)

    sub_data = pd.DataFrame({'AACombo': SEQ_list, 'Fitness': Fit_list,'Cluster': Cluster_list})
    trainingdata=os.path.join(save_dir , 'InputValidationData.csv')
    sub_data.to_csv(trainingdata, index=False)

    np.savez(os.path.join(save_dir, 'clustering.npz'), tree=tree)
    return trainingdata
def main_sampling(seed,args,save_dir):
    np.random.seed(seed)
    if args.input_path is None:
        input_path='Input/'+args.dataset+'/'
    else:
        input_path=args.input_path
    if not os.path.exists(save_dir):
        os.system('mkdir -p '+save_dir)
    groundtruth_file=os.path.join(input_path, args.dataset+'.xlsx')
    groundtruth = pd.read_excel(groundtruth_file)
    Fitness = groundtruth['Fitness'].values
    Fitness = Fitness / Fitness.max()

    # get feature matrix
    encoding_lib = os.path.join(input_path, args.dataset+'_'+args.encoding_ev + '.npy')
    features_zeroshot = np.load(encoding_lib)
    encoding_lib = os.path.join(input_path, args.dataset+'_'+args.encoding + '_normalized.npy')
    features = np.load(encoding_lib)
    ComboToIndex=pickle.load(open(os.path.join(input_path, 'ComboToIndex_'+args.dataset +'.pkl'),'rb'))
    AACombo = groundtruth['Variants'].values

    if len(features.shape) == 3:
        features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
    if len(features_zeroshot.shape) == 3:
        features_zeroshot = np.reshape(features_zeroshot, [features_zeroshot.shape[0], features_zeroshot.shape[1] * features_zeroshot.shape[2]])
    trainingdata=cluster_sample(args,save_dir,features,features_zeroshot,AACombo, Fitness,ComboToIndex)
    return trainingdata



if __name__ == "__main__":

    import argparse

    from time import  strftime
    time = strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("K_increments", nargs="+", help = "Increments of clusters at each hierarchy; Input a list; For example: --K_increments 10 0 10 10.")
    parser.add_argument("--K_zeroshot", help = "number of clusters divided by zeroshot embedding",type=int,default=10)
    parser.add_argument("--dataset", help = "Name of the data set. Options: 1. GB1; 2. PhoQ.", default = 'GB1')
    parser.add_argument("--encoding_ev", help = "encoding method used for initial sampling; Default: zero", default = 'zero')
    parser.add_argument("--encoding", help = "encoding method used for late-stage sampling and supervised model; Option: 1. AA; 2. Georgiev. Default: AA", default = 'AA')
    parser.add_argument("--num_first_round", help = "number of variants in the first round sampling; Default: 96",type=int,default=96)
    parser.add_argument("--batch_size", help = "Batch size. Number of variants can be screened in parallel. Default: 96",type=int,default = 96)
    parser.add_argument("--hierarchy_batch", help = "Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected, until max hierarchy. Default: 96",default = 96)
    parser.add_argument("--num_batch", help="number of batches; Default: 4",type=int,default=4)
    parser.add_argument('--input_path',help="Input Files Directory. Default 'Input/'",default=None)
    parser.add_argument('--save_dir', help="Output Files Directory; Default: current time", default= time + '/')
    parser.add_argument('--seed', help="random seed",type=int, default= 100)
    # parser.add_argument('--acquisition',help="Acquisition function used for in-cluster sampling; default UCB. Options: 1. UCB; 2. epsilon; 3. Thompson; 4. random. Default: random",default='random')
    # parser.add_argument('--sampling_para', help="Float parameter for the acquisition function. 1. beta for GP-UCB; 2. epsilon for epsilon greedy; 3&4. redundant for Thompson and random sampling. Default: 4.0",type=float, default= 4.0)
    parser.add_argument('--softmax_beta', help="The base of softmax is taken as exp(softmax_beta*z). It is used for sampling probability at the initial round using zeroshot predictions to focus on top clusters. Larger values tend to focus the top clusters",type=float, default= 10.0)
    # parser.add_argument('--threshold', help="Threshold for cluster sampling probability. The cluster sampling probability is calculated a normalized value of the average fitness in the clusters. If the probability is below this parameter, it will be set to be zero.",type=float, default=0.0)

    # parser.add_argument('--use_zeroshot',help="Whether to employ zeroshot predictor in sampling. Default: FALSE",type=bool, default=False)
    # parser.add_argument('--zeroshot',help="name of zeroshot predictor; Required a CSV file stored in directory $INPUT_PATH with name: $DATA_SET_zeroshot.csv. Default: EvMutation",default='EvMutation')
    # parser.add_argument('--N_zeroshot',help="Number of top ranked variants from zeroshot predictor used for the recombined library. Default: 1600",type=int,default=1600)

    args = parser.parse_args()


    # random seed for reproduction
    seed=args.seed
    main_sampling(seed,args,args.save_dir)

