from datasets import load_from_disk, DatasetDict
import math

data_dir = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter3_armo_dpo_abl/pref_filtered"
ds = load_from_disk(data_dir)      # 这是 DatasetDict
train = ds["train"]                # 取出 train split

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

filtered_train = train.filter(is_good)
print("原始数量:", len(train), "过滤后:", len(filtered_train))

# 🔴 关键：重新包成 DatasetDict 再保存
new_ds = DatasetDict({"train": filtered_train})
new_ds.save_to_disk(data_dir + "2")  # 会得到 dataset_dict.json + train/
