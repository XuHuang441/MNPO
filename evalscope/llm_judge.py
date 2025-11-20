from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy
import os
import argparse
from openai import OpenAI

api_key = os.getenv("OPEN_ROUTER_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--datasets", type=str, default="alpaca_eval")
    return parser.parse_args()

def main():
    args = parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(
        api_key="EMPTY",          # vLLM 默认无鉴权，给个占位即可
        base_url=base_url,
    )

    # 这里用一个最简单的 demo：对几个 prompt 生成回答
    prompts = [
        "Explain what is Reinforcement Learning from Human Feedback in one paragraph.",
        "Briefly compare PPO and DPO for aligning large language models.",
    ]

    for i, p in enumerate(prompts):
        resp = client.chat.completions.create(
            model=args.model,  # 要和 --served-model-name 保持一致
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        print(f"=== Sample {i} ===")
        print("Prompt:", p)
        print("Answer:", resp.choices[0].message.content)
        print()

if __name__ == "__main__":
    main()


task_cfg = TaskConfig(
    model='deepseek/deepseek-chat-v3.1',
    api_url='https://openrouter.ai/api/v1',
    api_key= api_key,
    eval_type=EvalType.SERVICE,
    datasets=[
        'arena_hard'
    ],
    eval_batch_size=12,
    judge_worker_num=12,
    # limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'gpt-5-nano',
        'generation_config': {"reasoning_effort": "minimal"},
        'api_url': 'https://openrouter.ai/api/v1',
        'api_key': api_key,
    },
    use_cache="/alloc/xu/evalscope/outputs/20250924_151921"
)


run_task(task_cfg=task_cfg)