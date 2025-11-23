import json
import random

input_path = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl"
out1 = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2_20k/gemma2_ufb_part2_split1.jsonl"
out2 = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2_20k/gemma2_ufb_part2_split2.jsonl"
out3 = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2_20k/gemma2_ufb_part2_split3.jsonl"

seed = 42  # 你的随机种子
random.seed(seed)

# 1. 读取所有行
with open(input_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# 2. 随机打乱
random.shuffle(lines)

# 3. 切分成 3 部分
n = len(lines)
part1 = lines[: n // 3]
part2 = lines[n // 3 : 2 * n // 3]
part3 = lines[2 * n // 3 : ]

# 4. 保存
def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

write_jsonl(out1, part1)
write_jsonl(out2, part2)
write_jsonl(out3, part3)

print("Done. Sizes:", len(part1), len(part2), len(part3))
