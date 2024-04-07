#!/bin/bash

#SBATCH --job-name='train_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q debug
            
#SBATCH -t 00-00:10:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

age_path="/data/amciilab/yiming/DATA/brain_age/masterdata.csv"
data_path="/data/amciilab/yiming/DATA/brain_age/extracted"

resume_checkpoint="./results/001-DiT-XL-16-3D/checkpoints/0004700.pt"

MODEL_FLAGS="--model DiT-XL/16 --resume-checkpoint $resume_checkpoint"

DATA_FLAGS="--data-path $data_path --age-path $age_path --num-classes 65 \
            --image-size 256 --in-channels 1\
            --global-batch-size 8 --epochs 8000 --num-workers 4"

SAMPLE_FLAGS="--labels 60\
              --ckpt-every 2000 --log-every 100"



master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

NUM_GPUS=2
torchrun --nproc-per-node $NUM_GPUS\
        --nnodes=1\
        --rdzv-backend=c10d\
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        train.py $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAGS