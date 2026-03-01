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

    def sample_list(self, u_values: list) -> list:
        return self.curve.evaluate_list(u_values)

    def get_derivatives(self, u: float, order: int) -> list:
        return self.curve.derivatives(u, order=order)

    def get_new_segment_parameter_range(self) -> tuple:
        """
        计算最新增加的一个控制点所对应的参数 u 的范围。
        基于均匀节点向量的性质： u_start = (n - p) / (n - p + 1)
        """
        n = self.num_ctrlpts - 1
        p = self.degree
        if n <= p:
            return 0.0, 1.0
        u_start = (n - p) / (n - p + 1)
        return u_start, 1.0


# --- 3. 环境主类 ---
class PipeRoutingEnv(gym.Env):
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
        self.SPLIT_STEP = 3
        self.max_steps = 60
        self.action_step_size = step_size
        self.sampling_interval_dl = 5.0  # 采样间隔

        # 传感器参数
        self.sensor_range = 200.0
        dir_list = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
        ]
        self.sensor_directions = [np.array(d) / np.linalg.norm(d) for d in dir_list]
        self.num_sensors = len(self.sensor_directions)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        single_obs_dim = 3 + 3 + 3 + self.num_sensors
        total_obs_dim = single_obs_dim * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        # 奖励权重
        self.weights = {
            'dist': 0.4,
            'len': 0.05,
            'obs': 0.2,
            'pe': 1.5,
            'success': 40.0,
            'curvature': 0.2,
            'torsion': 0.02,
            'flow_balance': 60.0
        }

        # 记录不同部分的曲线长度
        self.curve_lengths = {
            'trunk': 0.0,
            'branch1': 0.0,
            'branch2': 0.0
        }

    def _query_potential(self, point_xyz) -> float:
        idx = np.floor((point_xyz - self.origin) / self.leaf_size).astype(int)
        if (np.any(idx < 0) or np.any(idx >= self.grid_shape)):
            return -1.0
        return self.pe_map[idx[0], idx[1], idx[2]]

    def _calculate_tee_geometry(self, trunk_end, trunk_tangent):
        """计算贴壁三通几何"""
        local_p3 = np.array([0.0, 0.0, 40])  # Inlet
        local_p1 = np.array([-28.8, 0.0, -28.8])  # Outlet 1
        local_p2 = np.array([28.8, 0.0, -28.8])  # Outlet 2

        t_global_z = trunk_tangent / (np.linalg.norm(trunk_tangent) + 1e-6)
        radial_vec = np.array([-trunk_end[0], -trunk_end[1], 0.0])
        r_norm = np.linalg.norm(radial_vec)
        if r_norm < 1e-3:
            radial_vec = np.array([0.0, 0.0, 1.0])
        else:
            radial_vec /= r_norm

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

        T_trans = trunk_end - R @ local_p3
        global_p1 = T_trans + R @ local_p1
        global_p2 = T_trans + R @ local_p2

        global_v1 = R @ np.array([-1.0, 0.0, -1.0])
        global_v2 = R @ np.array([1.0, 0.0, -1.0])

        vec_to_t1 = self.targets[0]['point'] - trunk_end
        if np.dot(global_v1, vec_to_t1) < np.dot(global_v2, vec_to_t1):
            return [global_p2, global_p1], [global_v2, global_v1]
        return [global_p1, global_p2], [global_v1, global_v2]

    def reset(self):
        self.current_step = 0
        self.current_phase = self.PHASE_TRUNK

        self.curve_lengths = {'trunk': 0.0, 'branch1': 0.0, 'branch2': 0.0}

        p0 = self.P_s
        p1 = self.P_s + self.N_s * self.pipe_radius * 2
        p2 = p1 + self.N_s * self.pipe_radius * 2

        self.trunk_points = [p0.tolist(), p1.tolist(), p2.tolist()]
        self.trunk_weights = [1.0, 1.0, 1.0]

        self.trunk_head = p2
        self.trunk_prev = p1

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

        return np.concatenate([head / SCALE, vec / SCALE, tan, np.array(lidar)])

    def _get_observation(self):
        if self.current_phase == self.PHASE_TRUNK:
            s1 = self._get_single_slot(self.trunk_head, self.trunk_prev, 0)
            s2 = self._get_single_slot(self.trunk_head, self.trunk_prev, 1)
            phase = np.array([0.0], dtype=np.float32)
        else:
            s1 = self._get_single_slot(self.branch1_head, self.branch1_prev, 0)
            s2 = self._get_single_slot(self.branch2_head, self.branch2_prev, 1)
            phase = np.array([1.0], dtype=np.float32)
        return np.concatenate([s1, s2, phase]).astype(np.float32)

    def _calc_reward(self, head, prev, target_idx, step_len, history_points, history_weights, branch_id):
        target = self.target_guide_points[target_idx]

        # 1. 距离奖励
        d_old = np.linalg.norm(target - prev)
        d_new = np.linalg.norm(target - head)
        r_dist = (d_old - d_new) * self.weights['dist']

        # 2. 长度惩罚
        r_len = -step_len * self.weights['len']

        # --- [关键修改开始] ---
        # 我们不再单独检查 head，而是将避障积分到曲线采样中

        r_obs = 0.0
        r_pe = 0.0
        r_geom = 0.0

        # 临时列表用于收集势能，计算平均值
        pe_samples = []

        current_seq = history_points + [head.tolist()]
        current_weights = history_weights

        # 只有点数足够构建3阶曲线时才计算
        if len(current_seq) >= 4 and len(current_seq) == len(current_weights):
            try:
                curve = Geomdl_NURBS(current_seq, current_weights, degree=3)
                new_total_length = curve.length

                last_length = self.curve_lengths[branch_id]
                segment_length = new_total_length - last_length
                self.curve_lengths[branch_id] = new_total_length

                # 决定采样点数
                if segment_length <= 1e-3:
                    num_samples = 0
                else:
                    # 保证至少采几个点，避免"跨越"障碍
                    # 建议：最少采 5 个点，或者按步长除以5
                    num_samples = max(5, int(np.ceil(segment_length / self.sampling_interval_dl)))

                if num_samples > 0:
                    u_start, u_end = curve.get_new_segment_parameter_range()
                    # 生成采样参数 u
                    u_values = np.linspace(u_start, u_end, num_samples + 1)[1:]

                    # 获取曲线上的物理坐标点
                    sample_points = curve.sample_list(u_values)

                    k_vals = []
                    t_vals = []

                    # [关键循环] 同时检测几何属性和物理避障
                    for i, u in enumerate(u_values):
                        # 1. 几何计算
                        derivs = curve.get_derivatives(u, order=3)
                        k = _calculate_curvature(derivs)
                        t = _calculate_torsion(derivs)
                        k_vals.append(k)
                        t_vals.append(abs(t))

                        # 2. 避障检测 (使用曲线上的点 pt，而不是控制点 head)
                        pt = sample_points[i]
                        pe = self._query_potential(pt)

                        if pe < -0.5:  # 遇到障碍物
                            # 累加惩罚：穿过得越长，扣分越多
                            # 这里参考单管路逻辑，直接累加负的 pe 值
                            r_obs += pe * self.weights['obs']
                        else:
                            # 收集正势能 (引导场)
                            pe_samples.append(pe)

                    # 计算几何平均值
                    if k_vals:
                        mean_k = np.mean(k_vals)
                        mean_t = np.mean(t_vals)
                        r_geom = -(self.weights['curvature'] * mean_k + self.weights['torsion'] * mean_t)

            except Exception as e:
                # 容错：如果曲线构建失败，至少检查一下 head 点，作为保底
                pe_head = self._query_potential(head)
                if pe_head < -0.5:
                    r_obs = -10.0  # 给予固定重罚
                pass
        else:
            # 如果点不够构建曲线（起步阶段），退化为检查 head
            pe = self._query_potential(head)
            if pe < -0.5:
                r_obs = -2.0 * self.weights['obs']
            else:
                pe_samples.append(pe)

        # 计算势能引导奖励 (取平均)
        if pe_samples:
            mean_pe = np.mean(pe_samples)
            r_pe = (mean_pe - self.max_potential) * self.weights['pe']

        # --- [关键修改结束] ---

        return r_dist + r_len + r_obs + r_pe + r_geom

    def _calc_terminal_penalty(self, points, weights):
        """
        [新增] 终端段专用检查：
        当管路连接成功并追加了固定点后，检查最后一段（u=0.9~0.99）是否存在急转弯。
        """
        penalty = 0.0
        try:
            full_curve = Geomdl_NURBS(points, weights, degree=3)
            # 专门采样最后一段
            u_values = np.linspace(0.9, 0.99, 20)

            k_vals = []
            t_vals = []

            for u in u_values:
                derivs = full_curve.get_derivatives(u, order=3)
                k_vals.append(_calculate_curvature(derivs))
                t_vals.append(abs(_calculate_torsion(derivs)))

            if k_vals:
                # 给予比过程惩罚更大的权重
                w_k = self.weights['curvature'] * 50.0
                w_t = self.weights['torsion'] * 30.0
                penalty = -(w_k * np.mean(k_vals) + w_t * np.mean(t_vals))

        except Exception:
            pass

        return penalty

    def _calc_hydraulic_resistance(self, points, weights):
        """
        [新增] 估算单根管路的流阻代理值 R
        物理模型: R = 沿程阻力(与长度成正比) + 局部阻力(与曲率积分成正比)
        """
        if len(points) < 4:
            return 0.0

        try:
            curve = Geomdl_NURBS(points, weights, degree=3)
            length = curve.length

            # 采样计算累积曲率 (代表弯头带来的二次流阻力)
            u_values = np.linspace(0.0, 1.0, 20)[1:-1]
            total_curvature = 0.0
            for u in u_values:
                derivs = curve.get_derivatives(u, order=2)
                k = _calculate_curvature(derivs)
                total_curvature += k

            # 系数 alpha 和 beta 可根据实际管径和流体雷诺数标定
            # 这里给出一个经验性的权重：假设1单位长度的阻力，相当于一定曲率的阻力
            alpha = 1.0
            beta = 50.0  # 局部弯曲对流阻的影响通常被显著放大

            resistance = alpha * length + beta * total_curvature
            return resistance
        except Exception:
            # 容错处理
            return 9999.0

    def step(self, action):
        self.current_step += 1
        reward = 0.0

        # === Phase 0: Trunk ===
        if self.current_phase == self.PHASE_TRUNK:
            act = action[0:4]
            d_xyz = act[0:3] * self.action_step_size
            w_real = max(0.1, 1.0 + act[3] * 0.5)

            p_new = self.trunk_head + d_xyz
            step_len = np.linalg.norm(d_xyz)

            temp_weights = self.trunk_weights + [w_real]

            r1 = self._calc_reward(p_new, self.trunk_head, 0, step_len, self.trunk_points, temp_weights, 'trunk')
            r2 = self._calc_reward(p_new, self.trunk_head, 1, step_len, self.trunk_points, temp_weights, 'trunk')
            reward = r1 + r2

            self.trunk_weights.append(w_real)
            self.trunk_prev = self.trunk_head
            self.trunk_head = p_new
            self.trunk_points.append(p_new.tolist())

            if self.current_step == self.SPLIT_STEP:
                trunk_tan = self.trunk_head - self.trunk_prev
                trunk_tan_norm = trunk_tan / (np.linalg.norm(trunk_tan) + 1e-6)
                D = self.pipe_radius * 2.0

                p3_ext1 = self.trunk_head + trunk_tan_norm * D
                p3_ext2 = self.trunk_head + trunk_tan_norm * 2.0 * D
                self.trunk_points.append(p3_ext1.tolist())
                self.trunk_points.append(p3_ext2.tolist())
                self.trunk_weights.extend([1.0, 1.0])

                try:
                    full_trunk = Geomdl_NURBS(self.trunk_points, self.trunk_weights)
                    self.curve_lengths['trunk'] = full_trunk.length
                except:
                    pass

                # starts, tangents = self._calculate_tee_geometry(self.trunk_head, trunk_tan)
                starts, tangents = self._calculate_tee_geometry(p3_ext2, trunk_tan)
                # Branch 1 Init
                b1_center = starts[0]
                b1_tan = tangents[0]
                b1_fixed_pts = [b1_center, b1_center + b1_tan * D, b1_center + b1_tan * 2.0 * D]
                self.branch1_points = [p.tolist() for p in b1_fixed_pts]
                self.branch1_weights = [1.0, 1.0, 1.0]
                self.branch1_head = b1_fixed_pts[-1]
                self.branch1_prev = b1_fixed_pts[-2]
                try:
                    c1 = Geomdl_NURBS(self.branch1_points, self.branch1_weights)
                    self.curve_lengths['branch1'] = c1.length
                except:
                    pass

                # Branch 2 Init
                b2_center = starts[1]
                b2_tan = tangents[1]
                b2_fixed_pts = [b2_center, b2_center + b2_tan * D, b2_center + b2_tan * 2.0 * D]
                self.branch2_points = [p.tolist() for p in b2_fixed_pts]
                self.branch2_weights = [1.0, 1.0, 1.0]
                self.branch2_head = b2_fixed_pts[-1]
                self.branch2_prev = b2_fixed_pts[-2]
                try:
                    c2 = Geomdl_NURBS(self.branch2_points, self.branch2_weights)
                    self.curve_lengths['branch2'] = c2.length
                except:
                    pass

                self.current_phase = self.PHASE_BRANCH
                if self._query_potential(self.trunk_head) > -0.5:
                    reward += 5.0

        # === Phase 1: Branch ===
        else:
            # Branch 1
            if not self.done1:
                act1 = action[0:4]
                w1_real = max(0.1, 1.0 + act1[3] * 0.5)
                temp_w1 = self.branch1_weights + [w1_real]

                p1_new = self.branch1_head + act1[0:3] * self.action_step_size
                len1 = np.linalg.norm(p1_new - self.branch1_head)

                reward += self._calc_reward(p1_new, self.branch1_head, 0, len1, self.branch1_points,
                                            temp_w1, 'branch1')

                self.branch1_weights.append(w1_real)
                self.branch1_prev = self.branch1_head
                self.branch1_head = p1_new
                self.branch1_points.append(p1_new.tolist())

                if np.linalg.norm(p1_new - self.target_guide_points[0]) < 100.0:
                    reward += self.weights['success']
                    self.done1 = True
                    # 终点处理
                    tgt = self.targets[0]['point']
                    nrm = self.targets[0]['normal']
                    D = self.pipe_radius * 2.0
                    self.branch1_points.extend([
                        (tgt - nrm * 2.0 * D).tolist(),
                        (tgt - nrm * D).tolist(),
                        tgt.tolist()
                    ])
                    self.branch1_weights.extend([1.0, 1.0, 1.0])

                    # [新增] 计算终端惩罚
                    term_penalty = self._calc_terminal_penalty(self.branch1_points, self.branch1_weights)
                    reward += term_penalty

            # Branch 2
            if not self.done2:
                act2 = action[4:8]
                w2_real = max(0.1, 1.0 + act2[3] * 0.5)
                temp_w2 = self.branch2_weights + [w2_real]

                p2_new = self.branch2_head + act2[0:3] * self.action_step_size
                len2 = np.linalg.norm(p2_new - self.branch2_head)

                reward += self._calc_reward(p2_new, self.branch2_head, 1, len2, self.branch2_points,
                                            temp_w2, 'branch2')

                self.branch2_weights.append(w2_real)
                self.branch2_prev = self.branch2_head
                self.branch2_head = p2_new
                self.branch2_points.append(p2_new.tolist())

                if np.linalg.norm(p2_new - self.target_guide_points[1]) < 100.0:
                    reward += self.weights['success']
                    self.done2 = True
                    # 终点处理
                    tgt = self.targets[1]['point']
                    nrm = self.targets[1]['normal']
                    D = self.pipe_radius * 2.0
                    self.branch2_points.extend([
                        (tgt - nrm * 2.0 * D).tolist(),
                        (tgt - nrm * D).tolist(),
                        tgt.tolist()
                    ])
                    self.branch2_weights.extend([1.0, 1.0, 1.0])

                    # [新增] 计算终端惩罚
                    term_penalty = self._calc_terminal_penalty(self.branch2_points, self.branch2_weights)
                    reward += term_penalty

                # 检查是否刚刚在这一步双双完成任务
        just_finished = (self.done1 and self.done2) and not (self.current_step >= self.max_steps)

        if just_finished:
                    # ===[新增核心逻辑: 终端流阻平衡清算] ===
                    # 1. 计算两根管路的最终流阻代理值
            R1 = self._calc_hydraulic_resistance(self.branch1_points, self.branch1_weights)
            R2 = self._calc_hydraulic_resistance(self.branch2_points, self.branch2_weights)

                    # 2. 计算流阻不平衡度 (归一化差异: 0表示完美平衡，越大越不平衡)
                    # 加上 1e-6 防止分母为 0
            imbalance_ratio = abs(R1 - R2) / (max(R1, R2) + 1e-6)

                    # 3. 转化为奖励：差异越小，惩罚越小（或者转换为正奖励）
                    # 这里采用惩罚机制：如果不平衡度达到 1.0 (一根极其长一根极短)，扣除大量分数
            r_flow_balance = - imbalance_ratio * self.weights['flow_balance']

            reward += r_flow_balance

                    # （可选）在控制台打印日志，方便你观察 RL 是否学到了平衡
                    # print(f"Target reached! R1: {R1:.1f}, R2: {R2:.1f}, Imbalance: {imbalance_ratio:.2%}, Penalty: {r_flow_balance:.1f}")

                # 下面这两行也必须左对齐！保证每个 step 必然有返回值
        done = (self.current_step >= self.max_steps) or (self.done1 and self.done2)
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    #python -m spinup.run plot /home/ljh/PycharmProjects/branchpipe/ppo_results