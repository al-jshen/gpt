#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=00:15:00

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# run script from above
# srun python3 train.py --config config.toml 
srun python3 finetune.py --config config_finetune.toml 
