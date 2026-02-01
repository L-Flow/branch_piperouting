#!/usr/bin/env python3
# test_branching.py

import torch
import numpy as np
import os
from pipe_env import PipeRoutingEnv

# --- 1. 配置路径 (请修改为你实际的路径) ---
# 确保这些路径与训练时完全一致
PE_MAP_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_leaflevel(1).npy"
META_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_meta(1).json"
MODEL_PATH = "ppo_results/branching_run1/pyt_save/model.pt"

# --- 2. 定义双目标任务参数 ---
START_PT = [462.84, 273.6, -448.6]
START_N = [0, 0, 1]

TARGET_LIST = [
    # Target 1 (左侧分支目标)
    {'point': [44, 537.5, -229], 'normal': [-1, 0, 0]},
    {'point': [502.84, -188, -109.83], 'normal': [-10, -17.32, 0.0]}

]

# --- 3. 环境构建 ---
env_fn = lambda: PipeRoutingEnv(
    pe_map_path=PE_MAP_FILE,
    meta_path=META_FILE,
    start_point=START_PT,
    start_normal=START_N,
    target_list=TARGET_LIST,
    step_size=75.0
)


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return

    print(f"--- 加载模型: {MODEL_PATH} ---")
    try:
        ac = torch.load(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 创建环境
    env = env_fn()
    obs = env.reset()
    done = False

    print("--- 开始生成分支管路 ---")

    step_count = 0
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        # 获取确定性动作 (Deterministic Action)
        with torch.no_grad():
            action = ac.pi._distribution(obs_tensor).mean.numpy()

        obs, reward, done, info = env.step(action)
        step_count += 1

        # 简单的进度打印
        if step_count % 10 == 0:
            print(f"Step {step_count}: Phase={env.current_phase}")

    print("--- 生成完成 ---")

    # --- 4. 提取并保存数据 ---
    # 直接读取 env 中记录的真实数据
    # 注意：在 pipe_env.py 中我们已经确保了 points 和 weights 是同步 append 的
    data_groups = {
        "trunk": (env.trunk_points, env.trunk_weights),
        "branch1": (env.branch1_points, env.branch1_weights),
        "branch2": (env.branch2_points, env.branch2_weights)
    }

    saved_count = 0
    for name, (pts, wts) in data_groups.items():
        if not pts:
            print(f"[{name}] 数据为空，跳过。")
            continue

        # 数据完整性校验
        # 理论上 len(pts) 应该等于 len(wts)，或者 pts 比 wts 多一个初始点（取决于实现细节）
        # 这里取两者的最小长度进行截断，保证一一对应
        min_len = min(len(pts), len(wts))
        pts_save = pts[:min_len]
        wts_save = wts[:min_len]

        if min_len < 4:
            print(f"[{name}] 点数不足 ({min_len})，无法构成有效曲线，跳过保存。")
            continue

        pts_file = f"final_{name}_points.txt"
        wts_file = f"final_{name}_weights.txt"

        np.savetxt(pts_file, np.array(pts_save), fmt="%.6f")
        np.savetxt(wts_file, np.array(wts_save), fmt="%.6f")
        print(f"[{name}] 已保存: {pts_file} (点数: {len(pts_save)}, 权重数: {len(wts_save)})")
        saved_count += 1

    if saved_count == 3:
        print("\n所有管路段数据已成功导出！现在可以运行可视化脚本了。")
    else:
        print(f"\n警告：只导出了 {saved_count}/3 段管路数据，请检查模型是否成功完成了任务。")


if __name__ == "__main__":
    run_evaluation()