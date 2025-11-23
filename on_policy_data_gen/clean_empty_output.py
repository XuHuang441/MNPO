from datasets import Dataset, DatasetDict
import os
import glob

# ---- 你的原始数据目录 ----
data_dir = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter2_armo_abl/pref"

# 1. 找到 arrow 文件（一般在 train/ 目录下）
arrow_files = glob.glob(os.path.join(data_dir, "**/*.arrow"), recursive=True)
if not arrow_files:
    raise FileNotFoundError(f"在 {data_dir} 下没有找到 .arrow 文件")

arrow_path = arrow_files[0]
print(f"使用的 arrow 文件: {arrow_path}")

# 2. 直接从 arrow 文件构建 Dataset（绕过损坏的 dataset_info）
ds = Dataset.from_file(arrow_path)
print(ds)

# 3. 定义过滤条件：如果 all_generated_responses 中有任意一个元素是空，则丢弃该行
def is_valid(row):
    # 这里认为 空 / 全空格 / 换行 都算「空」
    return not any((r.strip() == "") for r in row["all_generated_responses"])

original_len = len(ds)
filtered = ds.filter(is_valid)

removed_count = original_len - len(filtered)
print(f"原始数量: {original_len}")
print(f"删除的行数: {removed_count}")
print(f"过滤后数量: {len(filtered)}")

# 4. 保存为新的 dataset（避免覆盖原始数据）
new_path = data_dir + "_filtered"
filtered_ds_dict = DatasetDict({"train": filtered})
filtered_ds_dict.save_to_disk(new_path)

print(f"过滤后的数据集已保存到: {new_path}")
