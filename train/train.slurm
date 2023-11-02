#!/bin/bash

#SBATCH --nodes=1             
#SBATCH --gpus-per-node=1     
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# run script from above
srun python3 train.py --dataset mnist
