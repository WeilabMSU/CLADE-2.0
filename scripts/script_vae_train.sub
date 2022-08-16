#!/bin/bash 
#SBATCH --time=0-23:59:00  
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
##SBATCH --mem-per-cpu= 128000M
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name vaetrain
#SBATCH --gres=gpu:1
#SBATCH --array=1-5 # job array index


source activate deep_sequence


dataset=GB1
THEANO_FLAGS='floatX=float32,device=cuda' python src/vae_train.py Input/"$dataset"/"$dataset".a2m "$dataset"_seed$[$SLURM_ARRAY_TASK_ID] $[$SLURM_ARRAY_TASK_ID]
