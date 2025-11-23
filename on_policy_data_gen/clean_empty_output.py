from datasets import load_from_disk, DatasetDict

data_dir = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/mnpo_iter2_armo_abl/pref"

ds = load_from_disk(data_dir)
train_ds = ds["train"]

# 定义一个过滤函数：如果all_generated_responses中任意元素为空字符串，就删除该行
def is_valid(row):
    # 返回 True 表示保留，False 表示删除
    # 判定条件：列表中不能出现 "" 或空白字符串
    return not any((r.strip() == "") for r in row["all_generated_responses"])

# 过滤数据
filtered = train_ds.filter(is_valid)

removed_count = len(train_ds) - len(filtered)
print(f"删除的行数: {removed_count}")
print(f"原始数量: {len(train_ds)}, 过滤后数量: {len(filtered)}")

# 保存新数据集
new_path = data_dir + "_filtered"
filtered_ds_dict = DatasetDict({"train": filtered})
filtered_ds_dict.save_to_disk(new_path)

print(f"过滤后的数据集已保存到: {new_path}")
