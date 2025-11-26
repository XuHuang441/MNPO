from datasets import load_from_disk

# ===== 1. 加载两个数据集 =====
ds1 = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/test_1_his")["train"]
ds2 = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/test_2_his")["train"]

# ===== 2. 你关注的 keys （两个数据集中都存在）=====
compare_keys = [
    'reference_chosen_logps',
    'reference_rejected_logps',
    'history0_chosen_logps',
    'history0_rejected_logps',
]

# ===== 3. 逐条比对 =====
def compare_datasets(ds1, ds2, keys):
    n1 = len(ds1)
    n2 = len(ds2)
    if n1 != n2:
        print(f"❌ 两个数据集长度不同: ds1={n1}, ds2={n2}")
        return

    mismatch_count = 0

    for idx in range(n1):
        row1, row2 = ds1[idx], ds2[idx]

        for k in keys:
            v1 = row1[k]
            v2 = row2[k]

            # 用 float 比较时注意 nan 的情况
            equal = (v1 == v2) or (isinstance(v1, float) and isinstance(v2, float) and (v1 != v1) and (v2 != v2))

            if not equal:
                mismatch_count += 1
                print(f"❌ 第 {idx} 行 key='{k}' 不同: ds1={v1}, ds2={v2}")

    if mismatch_count == 0:
        print("✅ 两个数据集在所有 ref 和 history0 的 logps 完全相同！")
    else:
        print(f"⚠️ 共发现 {mismatch_count} 处不一致")

# 运行对比
compare_datasets(ds1, ds2, compare_keys)


