#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p hai-lo
#SBATCH --gres=gpu:8
#SBATCH --job-name=mpo_train
#SBATCH --output=output_armo.log

set -e

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m mnpo_scripts.run_mnpo \
    "../training_configs/gemma-2-9b-it-mnpo-iter1-armo.yaml"