#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

data_path="/data/amciilab/yiming/DATA/brain_age/extracted"
age_path="/data/amciilab/yiming/DATA/brain_age/masterdata.csv"

dim=3
pos_embed_dim=4
steps=0018000
save=True

DiT_checkpoint="./results/009-DiT-L-2-ldm-3D/checkpoints/${steps}.pt"
vae_checkpoint="./logs_vae/32x64x64x128_IXI/checkpoints/checkpoint_160000.pt"

for num_noise_steps in 20 40 
do
        for cfg_scale in 2.0
        do
                log_path="./logs_new/009-DiT-L-2-ldm3d-${cfg_scale}-${num_noise_steps}-${steps}"

                PATH_FLAGS="--log-path $log_path --data-path $data_path --age-path $age_path\
                            --DiT-checkpoint $DiT_checkpoint --vae-checkpoint $vae_checkpoint"

                MODEL_FLAGS="--model DiT-L/2 --pos-embed-dim $pos_embed_dim --dim $dim --in-channels 3"

                
                DATA_FLAGS="--prefix IXI --num-batches 10 --batch-size 1 --image-size 28"
                SAMPLE_FLAG="--num-noise-steps $num_noise_steps --cfg-scale $cfg_scale --from-noise False\
                                --save $save"

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
                        ./scripts/translation.py $DATA_FLAGS $MODEL_FLAGS $SAMPLE_FLAG $PATH_FLAGS
        done
done