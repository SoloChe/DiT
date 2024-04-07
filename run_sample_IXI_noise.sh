#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-02:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

log_path="./logs/002-DiT-XL-16_noise"

MODEL_FLAGS="--model DiT-XL/16 --ckpt ./results/002-DiT-XL-16/checkpoints/0095000.pt"
DATA_FLAGS="--labels 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55\
            --num-classes 65 --image-size 256 --in-channels 1 --from-noise True"

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
        translation.py --log-dir $log_path $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAG