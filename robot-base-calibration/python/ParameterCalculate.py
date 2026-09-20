# -*- coding: utf-8 -*-
"""
ParameterCalculate.py  ——  三维刚体配准工具（现代 NumPy 实现）

说明：
    本版本已移除旧代码中的 np.mat / np.tile 等已废弃用法
    （np.mat 在 NumPy 2.0 中被移除），全部改用：
        - np.asarray(...)          代替 np.mat(...)
        - @ 运算符                  代替矩阵对象的 * 乘法
        - A.mean(axis=0)           代替 mean(A, axis=0)
        - 广播减法 A - mu          代替 np.tile(mu, (N, 1))
    兼容 NumPy 1.x 与 2.x。

功能：
    1. rigid_transform_3D      : SVD 求两组对应三维点之间的最优刚体变换 (R, t)
    2. rotation_matrix_to_euler: 旋转矩阵 -> 欧拉角（XYZ 与 KUKA OAT 两套，含万向锁处理）
    3. quaternion_calculator   : 旋转矩阵 -> 四元数 (w, x, y, z)，Shepperd 方法 + 归一化
    4. error_calculation       : 配准残差（残差向量 / 逐点范数 / RMSE / MAE）
    5. calculate               : 一站式调用（单一刚体变换模型）
    6. parse_robot_data / parse_base_data : KUKA(.dat) / FANUC(.ls) / ABB(.mod) 点位解析
    7. parse_robot_data_with_pose : 同时解析工具姿态 A/B/C（FANUC 的 W/P/R、ABB 的四元数）
    8. rigid_transform_with_tcp   : 9 参数模型 —— 连未知「电极帽 TCP 偏移 c」一起解出
    9. calculate_with_tcp         : 上述模型的单站调用

约定：
    - 变换方向  target ≈ R @ source + t（列向量）
    - R 为 3x3 正交矩阵，det(R) = +1（纯旋转，不含镜像）
    - 四元数顺序 (w, x, y, z)，角度单位为度

依赖：
    numpy
"""

import math
import re

import numpy as np


# ============================================================
# 1. 刚体配准：SVD 求最优 R, t
# ============================================================
def rigid_transform_3D(source, target):
    """
    求解两组对应三维点之间的最优刚体变换 (R, t)，使得
        target ≈ R @ source + t
    目标函数：
        min_{R,t}  sum_i || R @ source_i + t - target_i ||^2
        约束：R^T R = I, det(R) = +1

    参数
    ----
    source : (N, 3) 源点集
    target : (N, 3) 目标点集，与 source 一一对应

    返回
    ----
    R : (3, 3) 旋转矩阵（正交，det = +1）
    t : (3,)   平移向量
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    assert source.shape == target.shape, "两组点集形状必须一致"
    assert source.ndim == 2 and source.shape[1] == 3, "点集必须是 (N, 3)"

    # 1.1 质心
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    # 1.2 去质心（广播减法，取代 np.tile）
    S = source - mu_s
    T = target - mu_t

    # 1.3 互协方差矩阵 H = S^T @ T
    H = S.T @ T

    # 1.4 SVD 分解，最优旋转 R = V @ U^T
    U, s, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 1.5 反射修正：剔除镜像，保证真旋转
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 1.6 平移：t = mu_t - R @ mu_s
    t = mu_t - R @ mu_s

    return R, t


# ============================================================
# 2. 旋转矩阵 -> 欧拉角
# ============================================================
def rotation_matrix_to_euler(R):
    """
    返回
    ----
    xyz : [Rx, Ry, Rz]   内旋 XYZ 约定（度）
    zyx : [O, A, T]      KUKA OAT 约定（度），含万向锁退化处理
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3), "旋转矩阵必须是 3x3"

    # XYZ 欧拉角（Tait-Bryan）
    Rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    Ry = math.degrees(math.atan2(-R[2, 0],
                                 math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    Rz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    xyz = [Rx, Ry, Rz]

    # KUKA OAT
    sinA = math.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)
    if sinA < 1e-12:
        # 万向锁：A ≈ 0°/180°，O 与 T 耦合，令 O = 0
        A = math.degrees(math.atan2(sinA, R[2, 2]))
        O = 0.0
        T = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
    else:
        A = math.degrees(math.atan2(sinA, R[2, 2]))
        sA = math.sin(math.radians(A))
        O = math.degrees(math.atan2(R[1, 2] / sA, R[0, 2] / sA))
        T = math.degrees(math.atan2(R[2, 1] / sA, -R[2, 0] / sA))

    return xyz, [O, A, T]


# ============================================================
# 3. 旋转矩阵 -> 四元数（Shepperd 方法 + 归一化）
# ============================================================
def quaternion_calculator(R):
    """旋转矩阵 -> 单位四元数 [w, x, y, z]。"""
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3), "旋转矩阵必须是 3x3"

    four_sq = [
        R[0, 0] + R[1, 1] + R[2, 2],   # 4w^2 - 1
        R[0, 0] - R[1, 1] - R[2, 2],   # 4x^2 - 1
        R[1, 1] - R[0, 0] - R[2, 2],   # 4y^2 - 1
        R[2, 2] - R[0, 0] - R[1, 1],   # 4z^2 - 1
    ]
    idx = four_sq.index(max(four_sq))    # 选最大分量，避免除以接近 0 的数
    biggest = math.sqrt(four_sq[idx] + 1.0) * 0.5
    mult = 4.0 * biggest

    if idx == 0:
        w = biggest
        x = (R[2, 1] - R[1, 2]) / mult
        y = (R[0, 2] - R[2, 0]) / mult
        z = (R[1, 0] - R[0, 1]) / mult
    elif idx == 1:
        w = (R[2, 1] - R[1, 2]) / mult
        x = biggest
        y = (R[1, 0] + R[0, 1]) / mult
        z = (R[0, 2] + R[2, 0]) / mult
    elif idx == 2:
        w = (R[0, 2] - R[2, 0]) / mult
        x = (R[1, 0] + R[0, 1]) / mult
        y = biggest
        z = (R[2, 1] + R[1, 2]) / mult
    else:
        w = (R[1, 0] - R[0, 1]) / mult
        x = (R[0, 2] + R[2, 0]) / mult
        y = (R[2, 1] + R[1, 2]) / mult
        z = biggest

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)               # 归一化，消除数值误差
    return q.tolist()


# ============================================================
# 4. 误差计算
# ============================================================
def error_calculation(source, target, R, t):
    """
    计算配准残差。变换方向：target ≈ R @ source + t

    返回
    ----
    err_vec   : (N, 3) 残差向量  target - (R @ source + t)
    per_point : (N,)   逐点残差范数
    rmse      : float  均方根误差
    mae       : float  平均绝对误差
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)

    pred = (R @ source.T).T + t
    err_vec = target - pred
    per_point = np.linalg.norm(err_vec, axis=1)
    rmse = float(np.sqrt(np.mean(per_point ** 2)))
    mae = float(np.mean(per_point))
    return err_vec, per_point, rmse, mae


# ============================================================
# 5. 点位文件解析（KUKA .dat / FANUC .ls / ABB .mod / 三坐标 .txt）
# ============================================================
def parse_robot_data(text, robot_type):
    """从机器人程序文本中解析 TCP 点位，返回 (N, 3) ndarray。"""
    data = []
    for line in text.splitlines():
        if robot_type in ("KUKA", "FANUC"):
            b = re.split(r"[\s{}=,]", line)
            if len(b) > 1 and b[1]:
                if robot_type == "KUKA" and b[1] == "E6POS":
                    data.append([float(b[5]), float(b[7]), float(b[9])])
                elif robot_type == "FANUC" and b[1] == "X":
                    data.append([float(b[5]), float(b[14]), float(b[22])])
        else:  # ABB: MoveL / MoveP
            a = line.split()
            if a and a[0] in ("MoveL", "MoveP"):
                tok = a[1].replace("]", "").replace("[", "")
                b = tok.split(",")
                if len(b) > 10:
                    data.append([float(b[0]), float(b[1]), float(b[2])])
    return np.array(data)


def parse_base_data(text):
    """解析三坐标测量文本，返回 (N, 3) ndarray。自动跳过表头。

    兼容两种列格式：
        P1,164.374,-36.673,215.159              # 点号,X,Y,Z
        5782.200550,-346.425733,1770.947319     # X,Y,Z（现场三坐标常见导出格式）
    """
    data = []
    for line in text.splitlines():
        b = [v.strip() for v in re.split(r"[,;\t]", line.strip()) if v.strip()]
        if len(b) < 3:
            continue
        vals = []
        for v in b[:4]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)           # 非数字列（表头 / 点号）
        if len(b) >= 4 and None not in vals[1:4]:
            data.append(vals[1:4])          # 点号,X,Y,Z
        elif None not in vals[0:3]:
            data.append(vals[0:3])          # X,Y,Z
    return np.array(data).reshape(-1, 3)


# ============================================================
# 5b. 姿态解析（自动标定电极帽 TCP 用）
# ============================================================
def rotation_from_abc(A, B, C):
    """KUKA A/B/C（绕 Z-Y'-X'' 依次旋转，单位度）-> 旋转矩阵 Rz·Ry·Rx。"""
    a, b, c = math.radians(A), math.radians(B), math.radians(C)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return np.array([[ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
                     [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
                     [-sb,     cb * sc,                cb * cc]])


def rotation_from_quat(q):
    """ABB 四元数 [q1,q2,q3,q4] = [w,x,y,z] -> 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def parse_robot_data_with_pose(text, robot_type):
    """解析机器人程序，返回 (点位 (N,3), 工具姿态 (N,3,3) 或 None)。

    姿态用于第 8/9 节的电极帽 TCP 自动标定；取不到姿态时第二个返回值为 None。
    """
    pos, rot = [], []
    if robot_type == "KUKA":
        for m in re.finditer(r"E6POS[^{]*\{(.*?)\}", text):
            kv = dict(re.findall(r"([A-Za-z]+\d*)\s*([-+]?[\d.]+)", m.group(1)))
            if all(k in kv for k in ("X", "Y", "Z")):
                pos.append([float(kv["X"]), float(kv["Y"]), float(kv["Z"])])
                if all(k in kv for k in ("A", "B", "C")):
                    rot.append(rotation_from_abc(float(kv["A"]), float(kv["B"]), float(kv["C"])))
    elif robot_type == "FANUC":
        for m in re.finditer(r"P\[\d+\]\s*\{(.*?)\}", text, re.S):
            body = m.group(1)
            kv = dict(re.findall(r"([XYZ])\s*=\s*([-+]?[\d.]+)", body))
            if all(k in kv for k in ("X", "Y", "Z")):
                pos.append([float(kv["X"]), float(kv["Y"]), float(kv["Z"])])
                ang = dict(re.findall(r"([WPR])\s*=\s*([-+]?[\d.]+)", body))
                if all(k in ang for k in ("W", "P", "R")):
                    # FANUC W/P/R 为固定角 XYZ：Rz(R)·Ry(P)·Rx(W)
                    rot.append(rotation_from_abc(float(ang["R"]), float(ang["P"]), float(ang["W"])))
    else:  # ABB: MoveL [[x,y,z],[q1,q2,q3,q4],[...],[...]], v200, fine, tool0;
        for line in text.splitlines():
            a = line.split()
            if a and a[0] in ("MoveL", "MoveP"):
                b = a[1].replace("]", "").replace("[", "").split(",")
                if len(b) > 7:
                    pos.append([float(b[0]), float(b[1]), float(b[2])])
                    rot.append(rotation_from_quat(b[3:7]))
    pos = np.array(pos).reshape(-1, 3)
    if rot and len(rot) == len(pos):
        return pos, np.array(rot)
    return pos, None


def orientation_spread(Rt):
    """各点工具 Z 轴两两夹角最大值（度）。用于判断 TCP 是否可辨识。"""
    z, m = Rt[:, :, 2], 0.0
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            m = max(m, math.degrees(math.acos(max(-1.0, min(1.0, float(z[i] @ z[j]))))))
    return m


# ============================================================
# 6. 电极帽 TCP 自动标定：R、t 与工具偏移 c 同时求解
# ============================================================
# 现场机器人程序里的 E6POS 记录的是「编程 TCP」，而三坐标测的是电极帽实际触点，
# 两者在工具坐标系下相差一个固定偏移 c：
#        实际触点 = rob_i + Rtool_i · c
#        配准关系 = R · base_i + t = 实际触点
# 未知量 9 个（R 3 + t 3 + c 3）。目标函数对 c 线性、对 R 非凸，
# 且「交替最小化」会停在鞍点，故采用「确定性多起点 + Gauss-Newton/LM」。
def _hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _exp_so3(w):
    """旋转向量 -> 旋转矩阵（罗德里格斯公式）。"""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + _hat(w)
    K = _hat(w / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _fib_dirs(n):
    """Fibonacci 球面均匀方向，用作 c 的多起点。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi),
                     np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)


def _lm_tcp(base, rob, Rt, R, t, c, iters=60):
    """Levenberg-Marquardt。每步 R <- R·exp([dw])，天然保持正交。"""
    lam, n = 1e-3, len(base)
    eye3 = np.eye(3)
    def resid(R, t, c):
        return ((R @ base.T).T + t - (rob + Rt @ c)).reshape(-1)
    r = resid(R, t, c)
    cost = float(r @ r)
    for _ in range(iters):
        Bh = np.zeros((n, 3, 3))
        Bh[:, 0, 1] = -base[:, 2]; Bh[:, 0, 2] =  base[:, 1]
        Bh[:, 1, 0] =  base[:, 2]; Bh[:, 1, 2] = -base[:, 0]
        Bh[:, 2, 0] = -base[:, 1]; Bh[:, 2, 1] =  base[:, 0]
        J = np.concatenate([-np.einsum("ij,njk->nik", R, Bh), -Rt,
                            np.broadcast_to(eye3, (n, 3, 3))], axis=2).reshape(3 * n, 9)
        JtJ = J.T @ J
        g = J.T @ r
        d = None
        for _ in range(40):
            try:
                d = np.linalg.solve(JtJ + lam * np.diag(np.maximum(np.diag(JtJ), 1e-12)), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            Rn = R @ _exp_so3(d[:3]); cn = c + d[3:6]; tn = t + d[6:9]
            rn = resid(Rn, tn, cn)
            k = float(rn @ rn)
            if k < cost:
                R, t, c, r, cost = Rn, tn, cn, rn, k
                lam = max(lam / 3.0, 1e-12)
                break
            lam *= 10.0
        else:
            break
        if np.linalg.norm(d) < 1e-14:
            break
    return R, t, c


def rigid_transform_with_tcp(source, target, Rt, ndir=24):
    """9 参数刚体配准：target_i ≈ R @ source_i + t - Rtool_i @ c。

    即 source 为三坐标基点、target 为机器人 E6POS，Rt 为各点工具姿态；
    返回 (R, t, c, rmse)。多起点取残差最小者，结果确定可复现。
    """
    scale = float(np.linalg.norm(target - target.mean(axis=0), axis=1).mean()) or 1.0
    starts = [np.zeros(3)]
    for rad in (0.5 * scale, 1.0 * scale):
        starts.extend(list(_fib_dirs(ndir) * rad))
    best = None
    for c0 in starts:
        R0, t0 = rigid_transform_3D(source, target + Rt @ c0)
        R, t, c = _lm_tcp(source, target, Rt, R0, t0, c0)
        e = ((R @ source.T).T + t) - (target + Rt @ c)
        v = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
        if best is None or v < best[0]:
            best = (v, R, t, c)
    return best[1], best[2], best[3], best[0]


def calculate_with_tcp(rob_data, base_data, Rt):
    """一站式调用（含电极帽 TCP 自动标定）。

    返回 (t, euler_xyz, euler_oat, quat, R, err_vec, per_point, rmse, mae, tcp)，
    前 9 项含义与 calculate() 完全一致，最后多返回工具坐标系下的 TCP 偏移 c。
    """
    if Rt is None:
        raise ValueError("缺少工具姿态（A/B/C / W/P/R / 四元数），无法自动标定 TCP")
    R, t, tcp, _ = rigid_transform_with_tcp(np.asarray(base_data, float),
                                            np.asarray(rob_data, float), Rt)
    target = np.asarray(rob_data, float) + Rt @ tcp      # 实际电极帽触点
    euler_xyz, euler_oat = rotation_matrix_to_euler(R)
    quat = quaternion_calculator(R)
    err_vec, per_point, rmse, mae = error_calculation(base_data, target, R, t)
    return t, euler_xyz, euler_oat, quat, R, err_vec, per_point, rmse, mae, tcp


# ============================================================
# 6. 一站式调用
# ============================================================
def calculate(rob_data, base_data):
    """
    输入机器人坐标系和基坐标系下的两组对应点，返回配准参数和误差。

    变换方向：rob_data ≈ R @ base_data + t

    返回
    ----
    t         : (3,)      平移
    euler_xyz : [Rx,Ry,Rz] XYZ 欧拉角（度）
    euler_oat : [O,A,T]    KUKA OAT 角（度）
    quat      : [w,x,y,z]  四元数
    R         : (3,3)      旋转矩阵
    err_vec   : (N,3)      残差向量
    per_point : (N,)       逐点误差范数
    rmse      : float
    mae       : float
    """
    R, t = rigid_transform_3D(base_data, rob_data)
    euler_xyz, euler_oat = rotation_matrix_to_euler(R)
    quat = quaternion_calculator(R)
    err_vec, per_point, rmse, mae = error_calculation(base_data, rob_data, R, t)
    return t, euler_xyz, euler_oat, quat, R, err_vec, per_point, rmse, mae
