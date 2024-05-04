#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
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




model_id="003"
model="UDiT"
model_size="B"
model_patch="16"
dim=2
prefix="IXI"
steps="0500000"
pos_embed_dim=4
save=True

for num_noise_steps in 50
do
        for cfg_scale in 2.0
        do
                model_name=$(echo $model | tr '/' '-')
                log_path="./logs_new/${model_id}-${model_name}-${model_size}-${model_patch}-${dim}D-${prefix}-${cfg_scale}-${num_noise_steps}-${steps}"
                echo $log_path

                MODEL_FLAGS="--model $model --pos-embed-dim $pos_embed_dim\
                                --DiT-checkpoint ./results/${model_id}-${model_name}-${model_size}-${model_patch}-${dim}D-${prefix}/checkpoints/${steps}.pt"

                DATA_FLAGS="--data-path $data_path --age-path $age_path --num-batches 20 --batch-size 224\
                                --image-size 224 --in-channels 1 --dim $dim"

                SAMPLE_FLAG="--num-noise-steps $num_noise_steps --cfg-scale $cfg_scale --from-noise False\
                                --log-path $log_path --save $save"

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
                        ./scripts/translation.py $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAG
        done
done