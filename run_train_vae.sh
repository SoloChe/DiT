#!/bin/bash

#SBATCH --job-name='train_vae'
#SBATCH --nodes=1    
#SBATCH --mem=64G                 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1
#SBATCH -p general                
#SBATCH -q debug
            
#SBATCH -t 00-00:05:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

~/.conda/envs/torch_base/bin/python ./scripts/train_encoder.py --batch_size 1