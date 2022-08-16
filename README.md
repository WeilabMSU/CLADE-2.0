# CLADE 2.0: evolution-driven cluster learning-assisted directed evolution

It is a upgraded version of cluster learning-assisted directed evolution (CLADE). Evolutionary scores are used to initiate sampling in CLADE 2.0. Later sampling iteratively uses the available data to update sampling probability and clustering architecture. The last step of CLADE 2.0 uses MLDE to exploit fitness. 

# Table of Contents  

- [Installment](#installment)
- [Usage](#usage)
  * [Evolutionary scores](#evolution)
  * [Clustering Sampling](#cluster-learning-sampling)
  * [CLADE2.0](#CLADE2.0)
  * [DEMO](#DEMO)
- [Sources](#sources) 
- [Reference](#reference) 

First download this repo. 

Then install [MLDE](https://github.com/fhalab/MLDE#building-an-alignment-for-msa-transformer) for supervised learning model:
```
cd CLADE/ 
git clone --recurse-submodules https://github.com/fhalab/MLDE.git`
```
Other packages required:
1. Python3.6 or later.
2. [scikit-learn](https://scikit-learn.org/stable/)
3. numpy
4. pandas
5. pickle

Input data for dataset `$dataset` needs to be stored in `Input/$dataset/`. They include:

1. `$dataset.xlsx`: data for sequences and their experimental fitness. 

2. `$dataset_$encoding.npy`: feature matrix that encodes the sequences. Current `encoding` options: `AA` and `Georgive` for physiochemical encoding, and `zero` for ensemble evolutionary score. 

3. `ComboToIndex_$dataset.pkl`: dictionary file map sequence to its id given in the order given in `$dataset.xlsx` file.

4. `Mldeparameter.csv`: parameters for MLDE, stored in `Input/`

## Evolutionary scores
1. [DeepSequence VAE](https://github.com/debbiemarkslab/DeepSequence) is implemented by python 2.7 using THEANO. Use conda environment `scirpts/deep_sequence.yml` to set up the environment. Script for training VAE model is given in `scripts/script_vae_train.sub`. Script for caculation ELBO score is given in `scripts/script_vae.sub`.

2. EVmutation and MSA Transformer are obtained from implementation in [MLDE](https://github.com/fhalab/MLDE#building-an-alignment-for-msa-transformer).

3. HMM scores can be generated using `scripts/script_hmm.sub`

4. ESM-1v score can be generated using `scripts/script_esm.sub`

## Clustering Sampling
`clustering_sampling.py` Use hierarchical clustering to generate training data. 
```python
$ python3 clustering_sampling.py --help
```
### Inputs
#### positional arguments:  
`K_increments` Increments of clusters at each hierarchy; Input a list; For example: --K_increments 10 0 10 10.\
#### optional arguments: 
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. \
`--encoding_ev ENCODING_EV` encoding method used for initial sampling; Default: zero"
`--encoding ENCODING`  encoding method used for late-stage sampling and supervised model; Option: 1. AA; 2. Georgiev. Default: AA \
`--num_first_round NUM_FIRST_ROUND` number of variants in the first round sampling; Default: 96  \
`--batch_size BATCH_SIZE` Batch size. Number of variants can be screened in parallel. Default: 96  \
`--hierarchy_batch HIERARCHY_BATCH` Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected until max hierarchy. Default:96  \
`--num_batch NUM_BATCH` number of batches; Default: 4  \
`--input_path INPUT_PATH`  Input Files Directory. Default 'Input/'  \
`--save_dir SAVE_DIR`   Output Files Directory; Default: current time  \

### Outputs:
`InputValidationData.csv`: Selected labeled variants. Training data for downstream supervised learning. Default will generate 384 labeled variants with batch size 96.
`clustering.npz`: Indecis of variants in each cluster.
### Examples:  
In our work in CLADE 2.0, we always set the second K_increment as 0. In that case, the first round and the second round of sampling are performed on the same clusters to enhance the accuracy. The sampling probabilities in first round are driven by evolutionary scores, and that in the second round are given by the labeled data fitness:
`python3 cluster_sampling.py 10 0 10 10`
## CLADE2.0
`CLADE2.py` Run full process of CLADE. Run `cluster_sampling.py` and downstream supervised learning (MLDE).

### Inputs
It requires the same positional and optional arguments with `cluster_sampling.py`. 

It has an additional optional argument:

`--mldepara MLDEPARA`   List of MLDE parameters. Default: MldeParameters.csv 
### Outputs:
In additional to three output files from `cluster_sampling.py`, there are 6 files output from MLDE package. The most important one is: `PredictedFitness.csv` showing predicted fitness of all variants in the combinatorial library. The variants with higher predicted fitness have higher priority to be screened.
### Examples:
`python3 CLADE2.py 10 0 10 10 --batch_size 96 --num_first_round 96 --hierarchy_batch 96 --num_batch 4`
## DEMO: uses a linear model for MLDE: 

`python3 CLADE.py 10 0 10 10 --batch_size 96 --num_first_round 96 --hierarchy_batch 96 --num_batch 4 --mldepara Demo_MldeParameters.csv`

# Sources
## GB1 dataset
GB1 dataset (`GB1.xlsx`) can be obtained from: [Wu, Nicholas C., et al. "Adaptation in protein fitness landscapes is facilitated by indirect paths." Elife 5 (2016): e16965.](https://elifesciences.org/articles/16965)
## PhoQ dataset
PhoQ dataset (`PhoQ.xlsx`) is owned by Michael T. Laub's lab. Please cite: [Podgornaia, Anna I., and Michael T. Laub. "Pervasive degeneracy and epistasis in a protein-protein interface." Science 347.6222 (2015): 673-677.](https://science.sciencemag.org/content/347/6222/673.abstract)
## MLDE packages and zero-shot predictions
The supervised learning package MLDE and zero-shot predictions can be found in: [Wittmann, Bruce J., Yisong Yue, and Frances H. Arnold. "Informed training set design enables efficient machine learning-assisted directed protein evolution." Cell Systems (2021).](https://www.cell.com/cell-systems/fulltext/S2405-4712(21)00286-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471221002866%3Fshowall%3Dtrue)
## CLADE package
It can be found [here](https://github.com/WeilabMSU/CLADE) with paper: [Qiu Yuchi, Jian Hu, Guo-Wei Wei, "Cluster learning-assisted directed evolution" Nature Computational Science (2021)](https://www.nature.com/articles/s43588-021-00168-y).
## Reference
This work is under review.
