from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy
import os

# python evalscope/api_eval.py

api_key = os.getenv("OPEN_ROUTER_API_KEY")

task_cfg = TaskConfig(
    model='openai/gpt-5', # Created Aug 7, 2025
    generation_config={"reasoning_effort": "minimal", "max_tokens": 4096},
    api_url='https://openrouter.ai/api/v1',
    api_key=api_key,
    eval_type=EvalType.SERVICE,
    datasets=[
        'arena_hard'
    ],
    eval_batch_size=12,
    judge_worker_num=12,
    # limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'gpt-5-mini',
        'generation_config': {"reasoning_effort": "minimal"},
        'api_url': 'https://openrouter.ai/api/v1',
        'api_key': api_key,
    },
    use_cache="/hai/scratch/fangwu97/xu/MNPO/outputs/20251119_212223"
)


run_task(task_cfg=task_cfg)