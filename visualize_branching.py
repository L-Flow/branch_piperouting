#!/usr/bin/env python3
# visualize_branching.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomdl import NURBS, utilities, operations

# 定义数据文件名
FILES = {
    "Trunk": ("final_trunk_points.txt", "final_trunk_weights.txt", "black"),
    "Branch1": ("final_branch1_points.txt", "final_branch1_weights.txt", "blue"),
    "Branch2": ("final_branch2_points.txt", "final_branch2_weights.txt", "red"),
}


def load_curve(pts_file, wts_file):
    try:
        ctrlpts = np.loadtxt(pts_file).tolist()
        weights = np.loadtxt(wts_file).tolist()
        return ctrlpts, weights
    except Exception as e:
        print(f"无法加载 {pts_file}: {e}")
        return None, None


def main():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print("--- 开始可视化 ---")

    all_points = []  # 用于自动调整视角比例

    for name, (p_file, w_file, color) in FILES.items():
        pts, wts = load_curve(p_file, w_file)
        if pts is None: continue

        # 创建 NURBS 曲线
        curve = NURBS.Curve()
        curve.degree = 3

        # 容错：点数不足无法生成3阶曲线
        if len(pts) < 4:
            print(f"警告: {name} 点数不足 ({len(pts)}), 绘制折线替代。")
            pts_np = np.array(pts)
            ax.plot(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], color=color, label=f"{name} (Raw)", linewidth=2,
                    linestyle='--')
            continue

        curve.ctrlpts = pts
        curve.weights = wts
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(pts))

        # 生成渲染点 (Eval Points)
        curve.delta = 0.01
        eval_pts = np.array(curve.evalpts)
        all_points.append(eval_pts)

        # 绘制曲线
        ax.plot(eval_pts[:, 0], eval_pts[:, 1], eval_pts[:, 2],
                color=color, label=f"{name} Path", linewidth=3)

        # 绘制控制点 (可选)
        ctrl_np = np.array(pts)
        ax.scatter(ctrl_np[:, 0], ctrl_np[:, 1], ctrl_np[:, 2], color=color, s=20, alpha=0.3)

    # --- 设置图表 ---
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Automatic Branching Pipe Layout')

    # 自动调整坐标轴比例 (Equal Aspect Ratio)
    if all_points:
        all_pts_concat = np.concatenate(all_points, axis=0)
        max_range = np.array([
            all_pts_concat[:, 0].max() - all_pts_concat[:, 0].min(),
            all_pts_concat[:, 1].max() - all_pts_concat[:, 1].min(),
            all_pts_concat[:, 2].max() - all_pts_concat[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_pts_concat[:, 0].max() + all_pts_concat[:, 0].min()) * 0.5
        mid_y = (all_pts_concat[:, 1].max() + all_pts_concat[:, 1].min()) * 0.5
        mid_z = (all_pts_concat[:, 2].max() + all_pts_concat[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    main()