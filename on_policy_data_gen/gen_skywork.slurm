#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo_infer_sky
#SBATCH --output=logs/output_iskywork.log
#SBATCH --error=logs/output_iskywork.err
#SBATCH --mem=128G


stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
    --data_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split3.jsonl \
    --model /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_skywork \
    --seeds 13 21 42 79 100 \
    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_skywork \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --num_gpu 2