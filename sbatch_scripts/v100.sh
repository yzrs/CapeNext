#!/bin/bash
#SBATCH --output=sbatch_scripts/v100_output.txt
#SBATCH --error=sbatch_scripts/v100_err.txt
#SBATCH --partition=v100
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
sh script/train_v100.sh

