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

batch_size=4
loss_type=3
prefix='all'
crop=True
val_freq=40
save_freq=5
model_depth=18
use_trans=False
oversample=False

logging_dir="./logs_res/res${model_depth}_loss${loss_type}_${prefix}_${use_trans}_${oversample}_norm"

FLAG="--model_depth $model_depth --batch_size $batch_size --loss_type $loss_type \
    --prefix $prefix --crop $crop --val_freq $val_freq --use_trans $use_trans\
    --save_freq $save_freq --logging_dir $logging_dir --oversample $oversample"

~/.conda/envs/torch_base/bin/python ./scripts/train_val_res3d.py $FLAG