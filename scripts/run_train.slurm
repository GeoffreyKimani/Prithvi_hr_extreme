#!/bin/bash --login
#SBATCH --job-name=hrx_prithvi_train
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hrx_prithvi_train_%j.out
#SBATCH --error=logs/hrx_prithvi_train_%j.err

module purge
module load apps/binapps/anaconda3/2024.10
conda activate prithvi-wxc

cd ~/scratch/prithvi_hr_extreme
export PYTHONPATH=$PWD:$PYTHONPATH

python -u training/train_hr_prithvi.py
