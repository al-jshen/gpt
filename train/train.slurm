#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# run script from above
srun python3 train.py --dataset mnist --task reconstruction --model vit --gpus 1 --epochs 10 --batch_size 512 --precision bf16-mixed --strategy ddp --data_dir /mnt/home/jshen/ceph/data/ml/ # --augment noise --noise 0.2 
