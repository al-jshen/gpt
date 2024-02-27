#!/bin/bash

#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH -C "h100|a100-80gb|a100"
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00

# run script from above
srun python3 train.py --config config.toml 
# srun python3 finetune.py --config config_finetune.toml 
