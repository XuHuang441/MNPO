# run_minerva_task.py
import os
import argparse
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    # 这个 key 是给 judge（OpenRouter GPT-5-mini）用的
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")

    task_cfg = TaskConfig(
        # 这里的 model 名字要和 vLLM 的 --served-model-name 一致
        model=args.model_name,
        generation_config={"max_tokens": 4096},

        # 指向本地 vLLM 的 OpenAI 兼容接口
        api_url=f"http://127.0.0.1:{args.port}/v1",
        # vLLM 默认不开启鉴权，这里给个占位就行（或者 None，看你 EvalScope 要不要非空）
        api_key="EMPTY",

        eval_type=EvalType.SERVICE,
        datasets=['arena_hard'],
        eval_batch_size=12,
        judge_worker_num=18,
        judge_strategy=JudgeStrategy.AUTO,

        # judge 仍然走 OpenRouter 的 GPT-5-mini
        judge_model_args={
            'model_id': 'gpt-4.1',
            # 'generation_config': {"reasoning_effort": "minimal"},
            'generation_config': {'max_tokens': 4096},
            'api_url': 'https://openrouter.ai/api/v1',
            'api_key': openrouter_api_key,
        },
        use_cache="/hai/scratch/fangwu97/xu/MNPO/outputs/20251124_152739"
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
