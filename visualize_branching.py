#!/usr/bin/env python3
# visualize_branching.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomdl import NURBS, utilities, operations
import os

# 定义输入数据文件名
FILES = {
    "Trunk": ("final_trunk_points.txt", "final_trunk_weights.txt", "black"),
    "Branch1": ("final_branch1_points.txt", "final_branch1_weights.txt", "blue"),
    "Branch2": ("final_branch2_points.txt", "final_branch2_weights.txt", "red"),
}


def load_curve(pts_file, wts_file):
    """加载控制点和权重"""
    try:
        # 检查文件是否存在
        if not os.path.exists(pts_file) or not os.path.exists(wts_file):
            print(f"文件缺失: {pts_file} 或 {wts_file}")
            return None, None

        ctrlpts = np.loadtxt(pts_file)
        weights = np.loadtxt(wts_file)

        # 处理单点情况 (loadtxt 读取单个数字时是 0-d array)
        if ctrlpts.ndim == 1 and len(ctrlpts) == 3:
            ctrlpts = [ctrlpts.tolist()]
            weights = [float(weights)]
        else:
            ctrlpts = ctrlpts.tolist()
            weights = weights.tolist()

        return ctrlpts, weights
    except Exception as e:
        print(f"加载错误 {pts_file}: {e}")
        return None, None


def save_xyz_for_solidworks(points, filename):
    """
    将离散点保存为 SOLIDWORKS 可识别的格式 (X Y Z)
    """
    try:
        # header="mm" 可以指定单位，但通常直接存数字，导入时选单位更灵活
        np.savetxt(filename, points, fmt="%.6f", delimiter="\t")
        print(f"  -> 已导出 SOLIDWORKS 文件: {filename} (点数: {len(points)})")
    except Exception as e:
        print(f"  -> 导出失败: {e}")


def main():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print("--- 开始处理及可视化 ---")

    all_points = []  # 用于自动调整视角比例

    for name, (p_file, w_file, color) in FILES.items():
        pts, wts = load_curve(p_file, w_file)
        if pts is None: continue

        print(f"[{name}] 处理中...")

        # 创建 NURBS 曲线对象
        curve = NURBS.Curve()
        curve.degree = 3

        # 容错：点数不足无法生成3阶曲线，仅保存折线
        if len(pts) < 4:
            print(f"  警告: 点数不足 ({len(pts)}), 将导出折线点。")
            eval_pts = np.array(pts)

            # 绘制折线
            ax.plot(eval_pts[:, 0], eval_pts[:, 1], eval_pts[:, 2],
                    color=color, label=f"{name} (Polyline)", linewidth=2, linestyle='--')

        else:
            curve.ctrlpts = pts
            curve.weights = wts
            curve.knotvector = utilities.generate_knot_vector(curve.degree, len(pts))

            # --- 生成密集点 (用于 CAD 导入) ---
            # delta 越小，点越密，曲线越光滑
            curve.delta = 0.005  # 相当于每个节点区间生成 200 个点
            eval_pts = np.array(curve.evalpts)

            # 绘制光滑曲线
            ax.plot(eval_pts[:, 0], eval_pts[:, 1], eval_pts[:, 2],
                    color=color, label=f"{name} Path", linewidth=3)

            # 绘制控制点 (半透明散点)
            ctrl_np = np.array(pts)
            ax.scatter(ctrl_np[:, 0], ctrl_np[:, 1], ctrl_np[:, 2], color=color, s=20, alpha=0.3)

        # --- 收集范围数据 ---
        all_points.append(eval_pts)

        # --- 导出到 SOLIDWORKS ---
        # 文件名例如: sw_Trunk.txt
        sw_filename = f"sw_{name}.txt"
        save_xyz_for_solidworks(eval_pts, sw_filename)

    # --- 设置图表显示 ---
    ax.legend()
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Pipe Layout (Exported to SW)')

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

    print("\n--- 完成！请使用 '插入 > 曲线 > 通过 XYZ 点的曲线' 将 sw_*.txt 导入 SOLIDWORKS ---")
    plt.show()


if __name__ == "__main__":
    main()