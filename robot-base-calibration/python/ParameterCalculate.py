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
    5. calculate               : 一站式调用
    6. parse_robot_data / parse_base_data : KUKA(.dat) / FANUC(.ls) / ABB(.mod) 点位解析

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
    """解析三坐标测量文本（逗号分隔：点号,X,Y,Z），返回 (N, 3) ndarray。"""
    data = []
    for line in text.splitlines():
        if len(line) > 10:
            b = re.split(r"[,]", line)
            if len(b) > 3 and b[1] and b[2] and b[3]:
                try:
                    data.append([float(b[1]), float(b[2]), float(b[3])])
                except ValueError:
                    pass  # 跳过表头行
    return np.array(data)


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
