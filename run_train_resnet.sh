#!/bin/bash

#SBATCH --job-name='train_res'
#SBATCH --nodes=1    
#SBATCH --mem=32G                 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

~/.conda/envs/torch_base/bin/python train_val_res3d.py --batch_size 5