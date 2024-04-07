#!/bin/bash

#SBATCH --job-name='sample_dit'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 02-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate torch_base

data_path="/data/amciilab/yiming/DATA/brain_age/preprocessed_data_256_10_IXI/"
age_path="/data/amciilab/yiming/DATA/brain_age/masterdata.csv"

for num_noise_steps in 500
do
        for cfg_scale in 1.5 2.0 2.5
        do
                log_path="./logs/002-DiT-XL-16-${cfg_scale}-${num_noise_steps}"
                MODEL_FLAGS="--model DiT-XL/16 --ckpt ./results/002-DiT-XL-16/checkpoints/0095000.pt"
                DATA_FLAGS="--data-dir $data_path --age-dir $age_path --num-patients 100 --batch-size 100\
                        --num-classes 65 --image-size 256 --in-channels 1"

                SAMPLE_FLAG="--num-noise-steps $num_noise_steps --cfg-scale $cfg_scale --from-noise False\
                                --log-dir $log_path"

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