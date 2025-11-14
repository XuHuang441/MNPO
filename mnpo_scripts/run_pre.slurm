#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo
#SBATCH --output=output.log
#SBATCH --mem=128G

set -e

history_args=""
    if [ ${#history_paths[@]} -gt 0 ]; then
        # The space at the end is important
        history_args="--history_paths ${history_paths[*]} "
    fi

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m mnpo_scripts.precompute \
    --run_name "mnpo_iter1_precompute" \
    --model_name_or_path /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/gemma2_ufb_part1_split1_skywork_scored.jsonl \
    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter1/pref \
    ${history_args} \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --sanity_check False
