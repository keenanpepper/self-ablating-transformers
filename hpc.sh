#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=100gb
#SBATCH --job-name=s-a-transformer-initial
## SBATCH --array=0-2

module load CUDA/12.1.1
module load IPython/8.14.0-GCCcore-12.3.0
source $HOME/venvs/self-ablation/bin/activate

python train.py