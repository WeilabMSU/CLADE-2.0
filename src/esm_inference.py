'''
Infers pseudo log likelihood approximations from ESM models.
codes are rewritten from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
'''

import argparse
from collections import defaultdict
import os
import pathlib

import numpy as np
import pandas as pd
import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, BatchConverter

from utils import read_fasta, save
from utils.esm_utils import PLLFastaBatchedDataset, PLLBatchConverter
Position_DIR={'GB1':['V39', 'D40', 'G41', 'V54'],
              'PhoQ':['A97', 'V98', 'S101', 'T102']
              }
criterion = torch.nn.CrossEntropyLoss(reduction='none')

esm_dict = {
    'esm1v_1': 'esm1v_t33_650M_UR90S_1', # use first of 5 models
    'esm1v_2': 'esm1v_t33_650M_UR90S_2', # use first of 5 models
    'esm1v_3': 'esm1v_t33_650M_UR90S_3', # use first of 5 models
    'esm1v_4': 'esm1v_t33_650M_UR90S_4', # use first of 5 models
    'esm1v_5': 'esm1v_t33_650M_UR90S_5', # use first of 5 models
    'esm1b': 'esm1b_t33_650M_UR50S',}
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

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument('dataset', type=str)
    # parser.add_argument("--positions", help = "AA indices to target",
    #                     nargs = "+", dest = "positions", default = None, type = str)
    parser.add_argument("--model",type=str,help="model, options: 1. esm1b; 2. esm1v",default="esm1b")
    parser.add_argument(
        "--toks_per_batch", type=int, default=4096, help="maximum batch size"
    )
    parser.add_argument("--msa_transformer", action="store_true")
    # parser.add_argument("--save_hidden", action="store_true", help="save rep")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    dataset = args.dataset
    wt_fasta_file='Input/'+dataset+'/'+dataset+'.fasta'
    fasta_file='Input/'+dataset+'/'+dataset+'_seqs.fasta'
    output_path='Input/'+dataset+'/'+args.model+'.npy'
    output_dir=pathlib.Path('Input/'+dataset+'/')
    model, alphabet = pretrained.load_model_and_alphabet(esm_dict.get(args.model))
    #batch_converter = alphabet.get_batch_converter()
    batch_converter = PLLBatchConverter(alphabet)
    mask_idx = torch.tensor(alphabet.mask_idx)
    padding_idx = torch.tensor(alphabet.padding_idx)

    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        mask_idx = mask_idx.cuda()
        print("Transferred model to GPU")

    wt = read_fasta(wt_fasta_file)[0]
    wt_data = [("WT", wt, -1)]
    _, _, wt_toks, _ = batch_converter(wt_data)
    dataset = PLLFastaBatchedDataset.from_file(fasta_file, wt)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=batch_converter, batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    repr_layers = [model.num_layers]   # extract last layer

    pll_diff = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks, mask_pos) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
                mask_pos = mask_pos.to(device="cuda", non_blocking=True)
                wt_toks = wt_toks.to(device="cuda", non_blocking=True)

            mask = torch.zeros(toks.shape, dtype=torch.bool, device=toks.device)
            row_idx = torch.arange(mask.size(0)).long()
            mask[row_idx, mask_pos] = True

            masked_toks = torch.where(mask, mask_idx, toks)
            if args.msa_transformer:
                masked_toks = torch.unsqueeze(masked_toks, 1)
            out = model(masked_toks, repr_layers=repr_layers,
                    return_contacts=False)
            logits = out["logits"]
            if args.msa_transformer:
               logits = torch.squeeze(logits, 1)
            logits_tr = logits.transpose(1, 2)  # [B, E, T]
            loss = criterion(logits_tr, toks)
            npll = torch.sum(mask * loss, dim=1).to(device="cpu").numpy()

            wt_toks_rep = wt_toks.repeat(toks.shape[0], 1)
            masked_wt_toks = torch.where(mask, mask_idx, wt_toks_rep)
            if args.msa_transformer:
                masked_wt_toks = torch.unsqueeze(masked_wt_toks, 1)
            out = model(masked_wt_toks, repr_layers=repr_layers,
                    return_contacts=False)
            logits = out["logits"]
            if args.msa_transformer:
               logits = torch.squeeze(logits, 1)
            logits_tr = logits.transpose(1, 2)  # [B, E, T]
            loss_wt = criterion(logits_tr, wt_toks_rep)
            npll_wt = torch.sum(mask * loss_wt, dim=1).to(
                    device="cpu").numpy()

            for i, label in enumerate(labels):
                pll_diff[label].append(npll_wt[i] - npll[i])

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp=[np.sum(v) for k, v in pll_diff.items()]

    pll_diff = {k: np.sum(v) for k, v in pll_diff.items()}
    df = pd.DataFrame.from_dict(pll_diff, columns=['pll'], orient='index')
    df.to_csv(os.path.join(output_dir, args.model+'_pll.csv'), index=True)
    np.save(output_path,tmp)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
