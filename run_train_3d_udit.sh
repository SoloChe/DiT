#!/bin/bash

#SBATCH --job-name='train_udit'
#SBATCH --nodes=1    
#SBATCH --mem=64G                 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:2
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

age_path="/data/amciilab/yiming/DATA/brain_age/masterdata.csv"
data_path="/data/amciilab/yiming/DATA/brain_age/extracted"

# resume_checkpoint="./results/001-DiT-XL-16-3D/checkpoints/0004700.pt"

MODEL_FLAGS="--model UDiT-B/16 --pos-embed-dim 4"

DATA_FLAGS="--data-path $data_path --age-path $age_path --num-classes 65 \
            --image-size 224 --in-channels 1 --dim 3\
            --global-batch-size 8 --epochs 8000 --num-workers 4"

SAMPLE_FLAGS="--labels 60\
              --ckpt-every 2000 --log-every 100"



master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

NUM_GPUS=2
export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nproc-per-node $NUM_GPUS\
        --nnodes=1\
        --rdzv-backend=c10d\
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT\
        ./scripts/train_dunetr.py $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAGS