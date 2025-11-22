#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo_infer
#SBATCH --output=logs/output_infer.log
#SBATCH --error=logs/output_infer.err
#SBATCH --mem=128G

# iter2
#stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#        --data_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split2.jsonl \
#        --model /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k \
#        --seeds 13 21 42 79 100 \
#        --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_armo \
#        --cache_dir /hai/scratch/fangwu97/xu/cache \
#        --num_gpu 2

# iter3
#stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#        --data_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split3.jsonl \
#        --model /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.85_eta0.005 \
#        --seeds 13 21 42 79 100 \
#        --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_armo \
#        --cache_dir /hai/scratch/fangwu97/xu/cache \
#        --num_gpu 2

stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
        --data_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split3.jsonl \
        --model /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio1_eta0.005  \
        --seeds 13 21 42 79 100 \
        --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_armo_abl \
        --cache_dir /hai/scratch/fangwu97/xu/cache \
        --num_gpu 4