#!/bin/bash

#SBATCH --job-name='train_res'
#SBATCH --nodes=1    
#SBATCH --mem=32G                 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 02-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

batch_size=16
loss_type=1
logging_dir="./logs_res/res18_all_loss$loss_type"

~/.conda/envs/torch_base/bin/python ./scripts/train_val_res3d.py --loss_type $loss_type --batch_size $batch_size --logging_dir $logging_dir