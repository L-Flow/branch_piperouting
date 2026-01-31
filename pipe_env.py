#!/usr/bin/env python3
# pipe_env.py

import gym
from gym import spaces
import numpy as np
import json
from geomdl import NURBS, utilities, operations


# --- 1. 几何计算辅助函数 ---
def _calculate_curvature(derivs: list) -> float:
    """计算曲率 k = |r' x r''| / |r'|^3"""
    c_prime = np.array(derivs[1])
    c_double_prime = np.array(derivs[2])
    norm_prime = np.linalg.norm(c_prime)
    if norm_prime < 1e-6:
        return 0.0

    cross = np.cross(c_prime, c_double_prime)
    return np.linalg.norm(cross) / (norm_prime ** 3 + 1e-9)


def _calculate_torsion(derivs: list) -> float:
    """计算挠率 t = (r' x r'') . r''' / |r' x r''|^2"""
    c_prime = np.array(derivs[1])
    c_double_prime = np.array(derivs[2])
    c_triple_prime = np.array(derivs[3])

    cross = np.cross(c_prime, c_double_prime)
    norm_sq = np.linalg.norm(cross) ** 2
    if norm_sq < 1e-6:
        return 0.0

    return np.dot(cross, c_triple_prime) / norm_sq


# --- 2. NURBS 包装类 ---
class Geomdl_NURBS:
    """NURBS 曲线封装类"""

    def __init__(self, ctrlpts: list, weights: list, degree: int = 3):
        self.degree = degree
        # 确保数据格式正确
        self.ctrlpts = [list(map(float, p)) for p in ctrlpts]
        self.weights = list(map(float, weights))
        self.num_ctrlpts = len(self.ctrlpts)

        # 容错：如果点数少于 degree+1，补充最后一个点
        if self.num_ctrlpts < self.degree + 1:
            needed = self.degree + 1 - self.num_ctrlpts
            for _ in range(needed):
                self.ctrlpts.append(self.ctrlpts[-1])
                self.weights.append(self.weights[-1])
            self.num_ctrlpts = len(self.ctrlpts)

        self.curve = NURBS.Curve()
        self.curve.degree = self.degree
        self.curve.ctrlpts = self.ctrlpts
        self.curve.weights = self.weights
        self.curve.knotvector = utilities.generate_knot_vector(self.degree, self.num_ctrlpts)

    @property
    def length(self) -> float:
        try:
            return operations.length_curve(self.curve)
        except:
            return 0.0


# --- 3. 环境主类 ---
class PipeRoutingEnv(gym.Env):
    """
    支持分支管路自动布局的强化学习环境 (Single-Agent, Dual-Channel)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 pe_map_path: str,
                 meta_path: str,
                 start_point: list,
                 start_normal: list,
                 target_list: list,
                 pipe_diameter: float = 10.0,
                 step_size: float = 75.0):
        super(PipeRoutingEnv, self).__init__()

        # 加载环境数据
        print(f"Loading PE map from {pe_map_path}...")
        self.pe_map = np.load(pe_map_path)
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.origin = np.array(self.meta['leaf_grid_origin'])
        self.leaf_size = float(self.meta['leaf_size'])
        self.grid_shape = self.meta['leaf_grid_shape']
        self.max_potential = np.max(self.pe_map)

        # 任务参数
        self.P_s = np.array(start_point, dtype=np.float32)
        norm_s = np.linalg.norm(start_normal)
        self.N_s = np.array(start_normal, dtype=np.float32) / (norm_s + 1e-6)

        # 处理多目标
        self.targets = []
        self.target_guide_points = []
        for t in target_list:
            pt = np.array(t['point'], dtype=np.float32)
            nm = np.array(t['normal'], dtype=np.float32)
            nm = nm / (np.linalg.norm(nm) + 1e-6)
            self.targets.append({'point': pt, 'normal': nm})
            self.target_guide_points.append(pt)

        self.pipe_radius = pipe_diameter / 2.0

        # 阶段控制参数
        self.PHASE_TRUNK = 0.0
        self.PHASE_BRANCH = 1.0
        self.SPLIT_STEP = 3  # 建议设为20，给主干足够的生长时间
        self.max_steps = 60  # 增加总步数以适应分支生长
        self.action_step_size = step_size

        # 传感器参数
        self.sensor_range = 200.0
        dir_list = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
        ]
        self.sensor_directions = [np.array(d) / np.linalg.norm(d) for d in dir_list]
        self.num_sensors = len(self.sensor_directions)

        # 定义动作空间 (8维)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # 定义观测空间 (39维: 19 + 19 + 1)
        single_obs_dim = 3 + 3 + 3 + self.num_sensors
        total_obs_dim = single_obs_dim * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        # 奖励权重
        self.weights = {
            'dist': 0.3,
            'len': 0.03,  # 稍微调大长度惩罚
            'obs': 3,
            'pe': 1.0,
            'success': 40.0,
            'curvature': 0.1,  # 曲率惩罚
            'torsion': 0.05  # 挠率惩罚
        }

    def _query_potential(self, point_xyz) -> float:
        idx = np.floor((point_xyz - self.origin) / self.leaf_size).astype(int)
        if (np.any(idx < 0) or np.any(idx >= self.grid_shape)):
            return -1.0
        return self.pe_map[idx[0], idx[1], idx[2]]

    def _calculate_tee_geometry(self, trunk_end, trunk_tangent):
        """计算贴壁三通几何"""
        # Local Definition
        local_p3 = np.array([25.0, 0.0, 23.5])  # Inlet
        local_p1 = np.array([50.0, 0.0, 0.0])  # Outlet 1
        local_p2 = np.array([0.0, 0.0, 0.0])  # Outlet 2

        t_global_z = trunk_tangent / (np.linalg.norm(trunk_tangent) + 1e-6)

        # 径向向量 (指向 Z 轴)
        radial_vec = np.array([-trunk_end[0], -trunk_end[1], 0.0])
        r_norm = np.linalg.norm(radial_vec)
        if r_norm < 1e-3:
            radial_vec = np.array([1.0, 0.0, 0.0])
        else:
            radial_vec /= r_norm

        # 构建旋转矩阵 R (Local Z -> Trunk Tangent)
        u_z = -t_global_z
        proj = radial_vec - np.dot(radial_vec, u_z) * u_z
        if np.linalg.norm(proj) < 1e-3:
            arbitrary = np.array([0, 0, 1]) if abs(u_z[2]) < 0.9 else np.array([0, 1, 0])
            u_y = np.cross(u_z, arbitrary)
        else:
            u_y = proj
        u_y /= (np.linalg.norm(u_y) + 1e-9)
        u_x = np.cross(u_y, u_z)

        R = np.stack([u_x, u_y, u_z], axis=1)

        # 计算全局坐标
        T_trans = trunk_end - R @ local_p3
        global_p1 = T_trans + R @ local_p1
        global_p2 = T_trans + R @ local_p2

        # 简单的出口方向向量 (Local X 轴方向)
        global_v1 = R @ np.array([1.0, 0.0, 0.0])
        global_v2 = R @ np.array([-1.0, 0.0, 0.0])

        # 目标匹配
        vec_to_t1 = self.targets[0]['point'] - trunk_end
        if np.dot(global_v1, vec_to_t1) < np.dot(global_v2, vec_to_t1):
            return [global_p2, global_p1], [global_v2, global_v1]

        return [global_p1, global_p2], [global_v1, global_v2]

    def reset(self):
        self.current_step = 0
        self.current_phase = self.PHASE_TRUNK

        # 初始化主干
        p0 = self.P_s
        p1 = self.P_s + self.N_s * self.pipe_radius * 2
        p2 = p1 + self.N_s * self.pipe_radius * 2

        self.trunk_points = [p0.tolist(), p1.tolist(), p2.tolist()]
        # 初始化主干权重
        self.trunk_weights = [1.0, 1.0, 1.0]

        self.trunk_head = p2
        self.trunk_prev = p1

        # 初始化分支为空
        self.branch1_head = None
        self.branch2_head = None
        self.branch1_weights = []
        self.branch2_weights = []

        self.done1 = False
        self.done2 = False

        return self._get_observation()

    def _get_single_slot(self, head, prev, target_idx):
        SCALE = 1000.0
        if head is None:
            return np.zeros(19, dtype=np.float32)

        tan = head - prev
        norm = np.linalg.norm(tan)
        tan = tan / (norm + 1e-6)

        target_pt = self.target_guide_points[target_idx]
        vec = target_pt - head

        lidar = []
        for d in self.sensor_directions:
            dist = 0.0
            for step in range(1, 11):
                chk = head + d * (self.sensor_range / 10.0 * step)
                if self._query_potential(chk) < -0.5:
                    break
                dist = float(step) * (self.sensor_range / 10.0)
            lidar.append(dist / self.sensor_range)

        return np.concatenate([
            head / SCALE,
            vec / SCALE,
            tan,
            np.array(lidar)
        ])

    def _get_observation(self):
        # 39维
        if self.current_phase == self.PHASE_TRUNK:
            s1 = self._get_single_slot(self.trunk_head, self.trunk_prev, 0)
            s2 = self._get_single_slot(self.trunk_head, self.trunk_prev, 1)
            phase = np.array([0.0], dtype=np.float32)
        else:
            s1 = self._get_single_slot(self.branch1_head, self.branch1_prev, 0)
            s2 = self._get_single_slot(self.branch2_head, self.branch2_prev, 1)
            phase = np.array([1.0], dtype=np.float32)

        return np.concatenate([s1, s2, phase]).astype(np.float32)

    def _calc_reward(self, head, prev, target_idx, step_len, history_points, history_weights):
        target = self.target_guide_points[target_idx]

        d_old = np.linalg.norm(target - prev)
        d_new = np.linalg.norm(target - head)
        r_dist = (d_old - d_new) * self.weights['dist']
        r_len = -step_len * self.weights['len']

        pe = self._query_potential(head)
        r_obs = 0.0
        r_pe = 0.0
        if pe < -0.5:
            r_obs = -2.0 * self.weights['obs']
        else:
            r_pe = (pe - self.max_potential) * self.weights['pe']

        r_geom = 0.0
        current_seq = history_points + [head.tolist()]
        current_weights = history_weights

        if len(current_seq) >= 4 and len(current_seq) == len(current_weights):
            try:
                curve = Geomdl_NURBS(current_seq, current_weights, degree=3)
                u_values = np.linspace(0.9, 0.99, 5)
                k_vals = []
                t_vals = []

                for u in u_values:
                    derivs = curve.curve.derivatives(u, order=3)
                    k_vals.append(_calculate_curvature(derivs))
                    t_vals.append(abs(_calculate_torsion(derivs)))

                mean_k = np.mean(k_vals) if k_vals else 0.0
                mean_t = np.mean(t_vals) if t_vals else 0.0

                r_geom = -(self.weights['curvature'] * mean_k + self.weights['torsion'] * mean_t)
            except Exception:
                pass

        return r_dist + r_len + r_obs + r_pe + r_geom

    def step(self, action):
        self.current_step += 1
        reward = 0.0

        # === Phase 0: Trunk ===
        if self.current_phase == self.PHASE_TRUNK:
            act = action[0:4]
            d_xyz = act[0:3] * self.action_step_size
            w_real = max(0.1, 1.0 + act[3] * 0.5)
            self.trunk_weights.append(w_real)

            p_new = self.trunk_head + d_xyz
            step_len = np.linalg.norm(d_xyz)

            r1 = self._calc_reward(p_new, self.trunk_head, 0, step_len, self.trunk_points, self.trunk_weights)
            r2 = self._calc_reward(p_new, self.trunk_head, 1, step_len, self.trunk_points, self.trunk_weights)
            reward = r1 + r2

            self.trunk_prev = self.trunk_head
            self.trunk_head = p_new
            self.trunk_points.append(p_new.tolist())

            # 检查分叉
            if self.current_step == self.SPLIT_STEP:
                trunk_tan = self.trunk_head - self.trunk_prev
                trunk_tan_norm = trunk_tan / (np.linalg.norm(trunk_tan) + 1e-6)
                D = self.pipe_radius * 2.0  # 管径

                # 1. 锁定主干末端 (Center + 1D + 2D)
                p3_ext1 = self.trunk_head + trunk_tan_norm * D
                p3_ext2 = self.trunk_head + trunk_tan_norm * 2.0 * D
                self.trunk_points.append(p3_ext1.tolist())
                self.trunk_points.append(p3_ext2.tolist())
                self.trunk_weights.extend([1.0, 1.0])

                # 2. 计算分支几何
                starts, tangents = self._calculate_tee_geometry(self.trunk_head, trunk_tan)

                # 3. 设置 Branch 1 (Start + 1D + 2D)
                b1_center = starts[0]
                b1_tan = tangents[0]
                b1_fixed_pts = [
                    b1_center,
                    b1_center + b1_tan * D,
                    b1_center + b1_tan * 2.0 * D
                ]
                self.branch1_points = [p.tolist() for p in b1_fixed_pts]
                self.branch1_weights = [1.0, 1.0, 1.0]
                self.branch1_head = b1_fixed_pts[-1]
                self.branch1_prev = b1_fixed_pts[-2]

                # 4. 设置 Branch 2 (Start + 1D + 2D)
                b2_center = starts[1]
                b2_tan = tangents[1]
                b2_fixed_pts = [
                    b2_center,
                    b2_center + b2_tan * D,
                    b2_center + b2_tan * 2.0 * D
                ]
                self.branch2_points = [p.tolist() for p in b2_fixed_pts]
                self.branch2_weights = [1.0, 1.0, 1.0]
                self.branch2_head = b2_fixed_pts[-1]
                self.branch2_prev = b2_fixed_pts[-2]

                self.current_phase = self.PHASE_BRANCH
                if self._query_potential(self.trunk_head) > -0.5:
                    reward += 5.0

        # === Phase 1: Branch ===
        else:
            # Branch 1
            if not self.done1:
                act1 = action[0:4]
                w1_real = max(0.1, 1.0 + act1[3] * 0.5)
                self.branch1_weights.append(w1_real)

                p1_new = self.branch1_head + act1[0:3] * self.action_step_size
                len1 = np.linalg.norm(p1_new - self.branch1_head)

                reward += self._calc_reward(p1_new, self.branch1_head, 0, len1, self.branch1_points,
                                            self.branch1_weights)

                self.branch1_prev = self.branch1_head
                self.branch1_head = p1_new
                self.branch1_points.append(p1_new.tolist())

                if np.linalg.norm(p1_new - self.target_guide_points[0]) < 100.0:
                    reward += self.weights['success']
                    self.done1 = True

                    # [修改]: 到达终点时，追加3个固定控制点
                    tgt = self.targets[0]['point']
                    nrm = self.targets[0]['normal']
                    D = self.pipe_radius * 2.0

                    # 倒推的顺序: 远 -> 近 -> 终点
                    # 这样才能保证最后进入目标时是平滑切入
                    p_ext2 = tgt - nrm * 2.0 * D
                    p_ext1 = tgt - nrm * D
                    p_final = tgt

                    self.branch1_points.append(p_ext2.tolist())
                    self.branch1_points.append(p_ext1.tolist())
                    self.branch1_points.append(p_final.tolist())
                    self.branch1_weights.extend([1.0, 1.0, 1.0])

            # Branch 2
            if not self.done2:
                act2 = action[4:8]
                # [关键修正] 使用 act2[3] 而不是 act2[7] (act2 是切片后的数组, 索引为 0~3)
                w2_real = max(0.1, 1.0 + act2[3] * 0.5)
                self.branch2_weights.append(w2_real)

                p2_new = self.branch2_head + act2[0:3] * self.action_step_size
                len2 = np.linalg.norm(p2_new - self.branch2_head)

                reward += self._calc_reward(p2_new, self.branch2_head, 1, len2, self.branch2_points,
                                            self.branch2_weights)

                self.branch2_prev = self.branch2_head
                self.branch2_head = p2_new
                self.branch2_points.append(p2_new.tolist())

                if np.linalg.norm(p2_new - self.target_guide_points[1]) < 100.0:
                    reward += self.weights['success']
                    self.done2 = True

                    # [修改]: 到达终点时，追加3个固定控制点
                    tgt = self.targets[1]['point']
                    nrm = self.targets[1]['normal']
                    D = self.pipe_radius * 2.0

                    p_ext2 = tgt - nrm * 2.0 * D
                    p_ext1 = tgt - nrm * D
                    p_final = tgt

                    self.branch2_points.append(p_ext2.tolist())
                    self.branch2_points.append(p_ext1.tolist())
                    self.branch2_points.append(p_final.tolist())
                    self.branch2_weights.extend([1.0, 1.0, 1.0])

        done = (self.current_step >= self.max_steps) or (self.done1 and self.done2)
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass