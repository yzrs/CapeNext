#!/bin/bash
#SBATCH --output=sbatch_scripts/l40s_output.txt
#SBATCH --error=sbatch_scripts/l40s_err.txt
#SBATCH --partition=l40s
#SBATCH -J tinySwinClip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=dcgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB

echo "======nvidia-smi=========="
nvidia-smi
echo "=========================="
export HF_ENDPOINT="https://hf-mirror.com"
sh script/train_l40s.sh

