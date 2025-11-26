
# iter2 armo
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=1 -m mnpo_scripts.precompute_new \
    --run_name "mnpo_iter2_precompute" \
    --model_name_or_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_beta10_ratio0.33_eta0.0075_td \
    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_armo_abl_scored.jsonl \
    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/test_1_his \
    --history_paths /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_beta10_ratio0.33_eta0.0075_td  \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --sanity_check True

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=1 -m mnpo_scripts.precompute_new \
    --run_name "mnpo_iter2_precompute" \
    --model_name_or_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_beta10_ratio0.33_eta0.0075_td \
    --ref_model /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --train_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_armo_abl_scored.jsonl \
    --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/test_2_his \
    --history_paths /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_beta10_ratio0.33_eta0.0075_td  \
    /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --sanity_check True