#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo_train_array
#SBATCH --output=logs/output_armo_%A_%a.log
#SBATCH --mem=128G
#SBATCH --array=0-2

# sbatch mnpo_scripts/train_array.slurm

set -e

CONFIGS=(
  "/hai/scratch/fangwu97/xu/MNPO/training_configs/gemma-2-9b-it-mnpo-iter2-skywork-3pl4.yaml"
  "/hai/scratch/fangwu97/xu/MNPO/training_configs/gemma-2-9b-it-mnpo-iter2-skywork-3pl5.yaml"
  "/hai/scratch/fangwu97/xu/MNPO/training_configs/gemma-2-9b-it-mnpo-iter2-skywork-3pl6.yaml"
)

export MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))
echo "Using MASTER_PORT: $MASTER_PORT"

CONFIG_PATH=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Using config: $CONFIG_PATH"

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m mnpo_scripts.run_mnpo \
    "$CONFIG_PATH"
