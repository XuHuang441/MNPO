from datasets import load_from_disk
import math

data_dir = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter3_armo_dpo_abl/pref_filtered"
ds = load_from_disk(data_dir)
train = ds["train"]

logp_keys = [
    'reference_chosen_logps',
    'reference_rejected_logps',
    'history0_chosen_logps',
    'history0_rejected_logps',
    'history1_chosen_logps',
    'history1_rejected_logps'
]

def is_good(row):
    for k in logp_keys:
        v = row[k]
        # 必须是 float 且不是 NaN
        if not isinstance(v, float) or math.isnan(v):
            return False
    return True

# ⚡️ 过滤掉 bad rows
filtered = train.filter(is_good)

print(f"原始数量: {len(train)}, 过滤后数量: {len(filtered)}")

# ⚠️ 若你希望覆盖原数据集
filtered.save_to_disk(data_dir + "2")
