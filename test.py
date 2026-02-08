import os
import sys
import torch

from vllm import LLM, SamplingParams


def main():
    model_path = "/hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-2b-it"

    # 可选：确保你真的分配/可见两张卡（Slurm 下通常由 CUDA_VISIBLE_DEVICES 控制）
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    print(f"CUDA_VISIBLE_DEVICES = {visible}")

    n = torch.cuda.device_count()
    print(f"torch.cuda.device_count() = {n}")
    for i in range(n):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if n < 2:
        print("ERROR: Need at least 2 visible GPUs for tensor_parallel_size=2.")
        sys.exit(1)

    # 核心：tensor_parallel_size=2
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",        # 你前面也是 bfloat16
        max_model_len=4096,      # 可按需调小/调大
        enable_prefix_caching=True,
    )

    prompts = [
        "Write a short haiku about GPUs.",
        "Explain tensor parallelism in one paragraph.",
    ]

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    outputs = llm.generate(prompts, sampling)

    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        print("\n" + "=" * 80)
        print(f"[PROMPT {i}] {prompts[i]}")
        print(f"[OUTPUT {i}] {text.strip()}")


if __name__ == "__main__":
    main()
