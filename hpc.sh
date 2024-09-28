#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=100gb
#SBATCH --job-name=lbl-sweep-0
#SBATCH --array=0-2

k_size=(2 4 8)

module load CUDA/12.1.1
module load IPython/8.14.0-GCCcore-12.3.0
source $HOME/venvs/self-ablation/bin/activate

python train.py --model_name lbl-sweep-0-k=${k_size[$SLURM_ARRAY_TASK_ID]} --k_attention ${k_size[$SLURM_ARRAY_TASK_ID]} --k_neurons ${k_size[$SLURM_ARRAY_TASK_ID]} --has_layer_by_layer_ablation_mask --loss_coeff_base 0.1 --loss_coeff_ablated 0.9 