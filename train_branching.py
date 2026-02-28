#!/usr/bin/env python3
# train_branching.py

import os
import numpy as np
from spinup import ppo_pytorch as ppo
from pipe_env import PipeRoutingEnv
from spinup.utils.mpi_tools import mpi_fork

# --- 1. 配置路径 ---
PE_MAP_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_leaflevel(3).npy"
META_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_meta(3).json"

if not (os.path.exists(PE_MAP_FILE) and os.path.exists(META_FILE)):
    print(f"Error: Environment files not found.")
    exit()

# --- 2. 定义双目标任务 ---
# 起点
# START_PT = [419.87, -335.75, -634.55]
# START_N = [10, 17.32, 0.0]
#
# TARGET_LIST = [
#     # Target 1 (左侧分支目标)
#     {'point': [505.34, 210.84, -512.15], 'normal': [-10, 13.77, 0]},
#     {'point': [471.04, -98.03, -946.4] , 'normal': [0, 0, -1]}
#
# ]
# START_PT = [462.84, 273.6, -448.6]
# START_N = [0, 0, 1]
#
# TARGET_LIST = [
#     # Target 1 (左侧分支目标)
#     {'point': [44, 537.5, -229], 'normal': [-1, 0, 0]},
#     {'point': [502.84, -188, -109.83], 'normal': [-10, -17.32, 0.0]}
#
# ]



START_PT = [471.04, -98.03, -946.4]
START_N = [0, 0, -1]

TARGET_LIST = [
    # Target 1 (左侧分支目标)
    {'point': [535.98, -75.53, -256.99], 'normal': [1, 11.55, 0]},
    {'point': [58.5, -552.25, -309.25] , 'normal': [1, 0, 0]}

]

'''
[419.87, -335.75, -634.55]  [10, 17.32, 0.0]
[505.34, 210.84, -512.15]   [10, -13.77, 0]
[471.04, -98.03, -946.4]     [0, 0, 1]

'''
""" 
[58.5, -552.25, -309.25]    [1, 0, 0]
[535.98, -75.53, -256.99]   [1, 11.55, 0]
[471.04, -98.03, -946.4]    [0, 0, -1]

"""
# START_PT = [58.5, -552.25, -309.25]
# START_N = [1, 0, 0]
#
# TARGET_LIST = [
#     # Target 1 (左侧分支目标)
#     {'point': [535.98, -75.53, -256.99], 'normal': [1, 11.55, 0]},
#     {'point': [471.04, -98.03, -946.4], 'normal': [0, 0, -1]}
#
# ]

# START_PT = [419.87, -335.75, -634.55]
# START_N = [10, 17.32, 0.0]
#
# TARGET_LIST = [
#     # Target 1 (左侧分支目标)
#     {'point': [505.34, 210.84, -512.15], 'normal': [-10, 13.77, 0]},
#     {'point': [471.04, -98.03, -946.4], 'normal': [0, 0, -1]}
#
# ]


# --- 3. 环境构建函数 ---
env_fn = lambda: PipeRoutingEnv(
    pe_map_path=PE_MAP_FILE,
    meta_path=META_FILE,
    start_point=START_PT,
    start_normal=START_N,
    target_list=TARGET_LIST, # 传入列表
    step_size=75.0
)

# --- 4. PPO 超参数 ---
# 增加网络容量以处理 39维输入和分叉逻辑
ac_kwargs = dict(
    hidden_sizes=[512, 512, 256]
)

logger_kwargs = dict(
    output_dir="ppo_results/branching_run1",
    exp_name="branching_ppo"
)

if __name__ == "__main__":
    mpi_fork(n=10)
    ppo(
        env_fn=env_fn,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=4000,
        epochs=75,          # 适当增加训练轮数
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=6e-5  ,           # 稍微调大学习率
        vf_lr=1e-3,
        lam=0.97,
        max_ep_len=40,
        logger_kwargs=logger_kwargs
    )