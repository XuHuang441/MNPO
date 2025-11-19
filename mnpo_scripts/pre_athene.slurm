#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=pre_athene
#SBATCH --output=output_ath.log
#SBATCH --mem=128G

# shortcut: sbatch mnpo_scripts/pre_athene.slurm

set -e

# iter1 athne
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m mnpo_scripts.precompute \
#    --run_name "mnpo_iter1_precompute" \
#    --model_name_or_path /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
#    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
#    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter1_athene_scored.jsonl \
#    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter1_athene/pref \
#    --cache_dir /hai/scratch/fangwu97/xu/cache \
#    --sanity_check False

# iter2
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m mnpo_scripts.precompute \
#    --run_name "mnpo_iter2_precompute" \
#    --model_name_or_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_athene_beta1_ratio0.33_eta0.005 \
#    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
#    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_athene_scored.jsonl \
#    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter2_athene/pref \
#    --history_paths /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k \
#    /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_skywork_inpo_iter1_20k \
#    --cache_dir /hai/scratch/fangwu97/xu/cache \
#    --sanity_check False

# iter3
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m mnpo_scripts.precompute \
    --run_name "mnpo_iter2_precompute" \
    --model_name_or_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_athene_beta1_ratio0.85_eta0.005_weights0.75-0.25 \
    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_athene_scored.jsonl \
    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter3_athene/pref \
    --history_paths /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k \
    /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_skywork_inpo_iter1_20k \
    /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.85_eta0.005 \
    /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_skywork_ratio0.85_eta0.01 \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --sanity_check False
