#!/bin/bash

#SBATCH --nodes=1             
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2   
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# run script from above
srun python3 train.py --dataset mnist --task reconstruction --gpus 2 --epochs 30 --batch_size 256 --precision bf16-mixed --strategy ddp --num_blocks 6 # --augment noise --noise 0.2 --load_ckpt ./lightning_logs/version_1009299/checkpoints/epoch=29-step=3540.ckpt
