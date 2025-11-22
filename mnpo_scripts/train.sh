CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info /mnt/data/.cache/conda/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m mnpo_scripts.run_mnpo \
    "/root/xu/MNPO/training_configs/gemma-2-9b-it-mnpo-iter2-armo-td3.yaml"
