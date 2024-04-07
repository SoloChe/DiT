#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q debug
            
#SBATCH -t 00-00:15:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

data_path="/data/amciilab/yiming/DATA/cifar10"

MODEL_FLAGS="--model DiT-S/2 --ckpt ./results/003-DiT-S-2/checkpoints/0038000.pt"
DATA_FLAGS="--num-classes 10 --image-size 32 --in-channels 3"



master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS\
        --nnodes=1\
        --rdzv-backend=c10d\
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        sample.py $DATA_FLAGS $MODEL_FLAGS