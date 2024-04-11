#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 01-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

data_path="/data/amciilab/yiming/DATA/brain_age/extracted"
age_path="/data/amciilab/yiming/DATA/brain_age/masterdata.csv"

dim=2
pos_embed_dim=2


for num_noise_steps in 50 100
do
        for cfg_scale in 1.5 2.0 2.5
        do
                log_path="./logs_new/002-DiT-XL-16-${dim}D-${cfg_scale}-${num_noise_steps}"

                MODEL_FLAGS="--model DiT-XL/16 --pos-embed-dim $pos_embed_dim\
                                --ckpt ./results/001-DiT-XL-16-2D/checkpoints/0190000.pt"

                DATA_FLAGS="--data-path $data_path --age-path $age_path --num-batches 20 --batch-size 112\
                        --num-classes 65 --image-size 224 --in-channels 1 --dim $dim"

                SAMPLE_FLAG="--num-noise-steps $num_noise_steps --cfg-scale $cfg_scale --from-noise False\
                                --log-path $log_path"

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
                        translation.py $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAG
        done
done