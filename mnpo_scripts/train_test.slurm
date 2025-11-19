#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo_train_test
#SBATCH --output=output_train_test.log
#SBATCH --mem=128G

set -e

# sbatch mnpo_scripts/train.slurm

# mnpo loss
ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m mnpo_scripts.run_mnpo \
    "/hai/scratch/fangwu97/xu/MNPO/training_configs/gemma-2-9b-it-mnpo-iter2-test.yaml"
