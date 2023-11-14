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

mkdir /tmp/ml/

cp -r /mnt/home/jshen/ceph/data/ml/MNIST /tmp/ml/

# run script from above
srun python3 train.py --dataset mnist --task classification --model vit --gpus 1 --epochs 25 --batch_size 512 --precision bf16-mixed --strategy ddp --data_dir /tmp/ml/ # --load_ckpt ./lightning_logs/version_2957682/checkpoints/epoch\=25-step\=3068.ckpt
