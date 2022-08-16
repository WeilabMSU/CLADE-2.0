'''
Infers average ELBO values from DeepSequence VAE models.
Based on open source code from DeepSequence repo.
'''

'''
Fix a bug in DeepSequence package to make the index correctly.
In helper.py, revise the code at two places: 

At line 175, define a new variable:
`
#YQ 03/13
self.uniprot_focus_col_to_focus_trim_idx \
    = {idx_col+int(start):i for i,idx_col in enumerate(np.sort(np.asarray(self.focus_cols)))}
`
around line 249 revise the code:
`
for pos,wt_aa,mut_aa in mutant_tuple_list:
    # YQ: 03/19
    assert mut_seq[self.uniprot_focus_col_to_focus_trim_idx[pos]]==wt_aa
    mut_seq[self.uniprot_focus_col_to_focus_trim_idx[pos]] = mut_aa
`
'''

import argparse
import numpy as np
import os
# import pathlib
from shutil import copyfile
import sys
import time
# from Bio import SeqIO
# import utils
import pandas as pd
WORKING_DIR=""  # Put in the deepsequence directory
assert len(WORKING_DIR)>1, 'Please enter directory for DeepSequence'
N_ELBO_SAMPLES=400

module_path = os.path.abspath(WORKING_DIR)
if module_path not in sys.path:
        sys.path.append(module_path)

from DeepSequence.model import VariationalAutoencoder
from DeepSequence import helper
from DeepSequence import train
offset_DIR={'GB1':1,
            'PhoQ':1}
model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
}
Position_DIR={'GB1':['V39', 'D40', 'G41', 'V54'],
              'PhoQ':['A97', 'V98', 'S101', 'T102']
              }
def generate_seqs(wt, positions, variants):
    for p in positions:
        assert wt[int(p[1:])-1]==p[0]
    seqs=[]
    for v in variants:
        seq=list(wt)
        for i,p in enumerate(positions):
            assert seq[int(p[1:])-1]==p[0]
            seq[int(p[1:])-1]=v[i]
        seqs.append("".join(seq))
    return seqs
def read_fasta(filename, return_ids=False):
    # records = SeqIO.parse(filename, 'fasta')
    # seqs = list()
    # ids = list()
    # for record in records:
    #     seqs.append(str(record.seq))
    #     ids.append(str(record.id))
    # if return_ids:
    #     return seqs, ids
    # else:
    #     return seqs
    seqs=list()
    ids = list()
    lines=open(filename,'r').readlines()
    seq=''
    for line in lines:
        if line[0]=='>':
            ids.append(line.replace('>','').replace('\n',''))
            if not seq=='':
                seqs.append(seq)
            seq=''
        else:
            seq=seq+line.replace('\n','')
            # if '\n' in seq:
            #     seqs.append(seq.replace('\n',''))
    seqs.append(seq)
    if return_ids:
        return seqs, ids
    else:
        return seqs
def seq2mutation_fromwt(seq, wt, ignore_gaps=False, sep=':', offset=1,
        focus_only=True):
    mutations = []
    for i in range(offset, offset+len(seq)):
        if ignore_gaps and ( seq[i-offset] == '-'):
            continue
        if wt[i-offset].islower() and focus_only:
            continue
        if seq[i-offset].upper() != wt[i-offset].upper():
            mutations.append((i, wt[i-offset].upper(), seq[i-offset].upper()))
    return mutations
def single_model(args,seed):
    dataset = args.dataset
    seq_path='Input/'+dataset+'/'+dataset+'.fasta'
    data_path='Input/'+dataset+'/'+dataset+'.xlsx'
    # output_path='Input/'+dataset+'/'+'elbo.npy'
    if seed==-1:
        output_path='Input/'+dataset+'/'+'elbo.npz'
    elif args.batch_id==-1:
        output_path='Input/'+dataset+'/'+'elbo_seed'+str(seed)+'.npz'
    else:
        output_path='Input/'+dataset+'/'+'elbo_seed'+str(seed)+'_batch'+str(args.batch_id)+'.npz'

    alignment_file='Input/'+dataset+'/'+dataset+'.a2m'
    wt = read_fasta(seq_path)#, return_ids=True)
    assert len(wt) == 1
    wt = wt[0]

    variants = pd.read_excel(data_path)['Variants'].values
    if args.positions is None:
        positions=Position_DIR.get(dataset)
    else:
        positions=args.positions
    seqs = generate_seqs(wt, positions, variants)
    # des = des[0]
    # offset = int(des.split('/')[-1].split('-')[0])
    offset=offset_DIR.get(dataset)
    data_helper = helper.DataHelper(
            working_dir=WORKING_DIR,
            alignment_file=alignment_file,
            calc_weights=False,
    )

    vae_model   = VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        working_dir                    =   WORKING_DIR,
        )
    if seed==-1:
        vae_model.load_parameters(dataset)
        # elbo_file='elbo.npy'
    else:
        vae_model.load_parameters(dataset+'_seed'+str(seed))
        # elbo_file='elbo_seed'+str(seed)+'.npy'


    # pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(args.output_dir).mkdir(parents=True)
    # os.system('mkdir '+output_dir)
    # focuscols = set(data_helper.uniprot_focus_cols_list)
    focuscols = data_helper.uniprot_focus_cols_list

    # seqs = read_fasta(args.fasta_file)

    # delta_elbos = np.zeros(len(seqs))
    delta_elbos = []
    VARIANTS=[]
    for i, s in enumerate(seqs):
        # if i % 100 == 0:
        #     # print(f'Computed elbos for {i} out of {len(seqs)} seqs')
        #     np.savetxt(os.path.join(args.output_dir, elbo_file), delta_elbos)
        if args.batch_id==-1:
            mut_tups = seq2mutation_fromwt(s, wt, offset=offset)
            mut_tups = [t for t in mut_tups if t[0] in focuscols]
            print(i)
            print(mut_tups)
            delta_elbos.append(data_helper.delta_elbo(vae_model, mut_tups,
                    N_pred_iterations=N_ELBO_SAMPLES))
        else:
            a=args.batch_id*args.batch_size
            b=(args.batch_id+1)*args.batch_size
            run_flag=False
            if (i>=a and i<b):
                run_flag=True
            elif b==len(seqs)-1 and i==b:
                run_flag = True
            if run_flag:
                mut_tups = seq2mutation_fromwt(s, wt, offset=offset)
                mut_tups = [t for t in mut_tups if t[0] in focuscols]
                print(i)
                print(mut_tups)
                delta_elbos.append(data_helper.delta_elbo(vae_model, mut_tups,
                        N_pred_iterations=N_ELBO_SAMPLES))
                VARIANTS.append(variants[i])
        # delta_elbos[i] = data_helper.delta_elbo(vae_model, mut_tups,
        #         N_pred_iterations=N_ELBO_SAMPLES)
    # np.savetxt(os.path.join(args.output_dir, elbo_file), delta_elbos)
    if len(delta_elbos)>0:
        if args.batch_id == -1:
            np.savez(output_path,elbos=delta_elbos)
        else:
            np.savez(output_path,elbos=delta_elbos,variants=VARIANTS)

    return delta_elbos

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("dataset", type=str)
    # parser.add_argument("fasta_file", type=str)
    # parser.add_argument("wt_fasta_file", type=str)
    # parser.add_argument("output_dir", type=str)
    parser.add_argument('dataset', type=str,help='wild type sequence in .fasta format')
    parser.add_argument("--positions", help = "AA indices to target",
                        nargs = "+", dest = "positions", default = None, type = str)

    parser.add_argument("--seed", type=int,default=-1)
    parser.add_argument("--batch_size", type=int,default=1000)
    parser.add_argument("--batch_id", type=int,default=-1)

    args = parser.parse_args()


    single_model(args,args.seed)

if __name__ == "__main__":
    main()
