from clustering_sampling import main_sampling
import os
import pandas as pd
def mlde(args,save_dir,trainingdata):
    input_path = 'Input/'
    encoding = args.encoding
    dataset = args.dataset
    encoding_lib = os.path.join(input_path, dataset+'/'+dataset+'_'+encoding+'_normalized.npy')
    combo_to_index =os.path.join(input_path,dataset+'/'+'ComboToIndex' + '_'+dataset + '.pkl')
    mldepara=os.path.join(input_path,args.mldepara)

    os.system('python3 MLDE/execute_mlde.py ' +trainingdata +' '+ \
              encoding_lib +' '+combo_to_index+' --model_params '+mldepara +' --output '+save_dir +' --hyperopt')
if __name__ == "__main__":

    import argparse
    from time import  strftime
    time = strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("K_increments", nargs="+", help = "Increments of clusters at each hierarchy; Input a list; For example: --K_increments 10 0 10 10.")
    parser.add_argument("--K_zeroshot", help = "number of clusters divided by zeroshot embedding",type=int,default=10)
    parser.add_argument("--dataset", help = "Name of the data set. Options: 1. GB1; 2. PhoQ.", default = 'GB1')
    parser.add_argument("--encoding_ev", help="encoding method used for initial sampling; Default: zero",
                        default='zero')
    parser.add_argument("--encoding",
                        help="encoding method used for late-stage sampling and supervised model; Option: 1. AA; 2. Georgiev. Default: AA",
                        default='AA')
    parser.add_argument("--num_first_round", help = "number of variants in the first round sampling; Default: 96",type=int,default=96)
    parser.add_argument("--batch_size", help = "Batch size. Number of variants can be screened in parallel. Default: 96",type=int,default = 96)
    parser.add_argument("--hierarchy_batch", help = "Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected, until max hierarchy. Default: 96",default = 96)
    parser.add_argument("--num_batch", help="number of batches; Default: 4",type=int,default=4)
    parser.add_argument('--input_path',help="Input Files Directory. Default 'Input/'",default=None)
    parser.add_argument('--save_dir', help="Output Files Directory; Default: current time", default= time + '/')
    parser.add_argument('--seed', help="random seed",type=int, default= 100)
    parser.add_argument('--softmax_beta', help="The base of softmax is taken as exp(softmax_beta*z). It is used for sampling probability at the initial round using zeroshot predictions to focus on top clusters. Larger values tend to focus the top clusters",type=float, default= 10.0)


    ## parameters for MLDE
    parser.add_argument("--mldepara",help="List of MLDE parameters; Default: MldeParameters.csv",default='MldeParameters.csv')

    args = parser.parse_args()


    # random seed for reproduction
    seed=args.seed
    dataset=args.dataset
    encoding=args.encoding
    # output_dir=args.save_dir
    args = parser.parse_args()


    trainingdata=main_sampling(seed,args,args.save_dir)

    mlde(args, save_dir,trainingdata)







