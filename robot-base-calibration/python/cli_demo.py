# -*- coding: utf-8 -*-
"""
cli_demo.py —— 命令行自测 / 演示

用法：
    python cli_demo.py                          # 内置真值自测
    python cli_demo.py robot.dat base.txt KUKA  # 计算真实数据
    python cli_demo.py robot.ls  base.txt FANUC
    python cli_demo.py robot.mod base.txt ABB
"""
import math
import sys

import numpy as np

from ParameterCalculate import (
    calculate, parse_robot_data, parse_base_data,
)


def euler_to_R(rx, ry, rz):
    """由 XYZ 欧拉角（度）构造旋转矩阵，用于生成测试真值。"""
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def self_test():
    np.set_printoptions(precision=6, suppress=True)
    print("=" * 64)
    print("内置自测：构造已知变换真值，验证算法恢复精度")
    print("=" * 64)

    # 真值：旋转 (5°, -8°, 12°) + 平移 (120, -85, 430) mm
    R_true = euler_to_R(5.0, -8.0, 12.0)
    t_true = np.array([120.0, -85.0, 430.0])

    rng = np.random.default_rng(42)
    base = rng.uniform(-300, 300, size=(7, 3)).round(3)      # 基坐标系点（三坐标）
    rob = (R_true @ base.T).T + t_true                        # 机器人坐标系点
    rob += rng.normal(0, 0.02, base.shape)                    # 0.02mm 测量噪声

    t, euler, oat, quat, R, err, per_point, rmse, mae = calculate(rob, base)

    print("\n[恢复精度]")
    print("  R 误差 (Frobenius) :", np.linalg.norm(R - R_true))
    print("  t 真值             :", t_true)
    print("  t 估计             :", np.round(t, 4))
    print("\n[输出参数]")
    print("  欧拉角 Rx Ry Rz (°):", np.round(euler, 4))
    print("  KUKA O  A   T  (°) :", np.round(oat, 4))
    print("  四元数 (w,x,y,z)   :", np.round(quat, 6))
    print("\n[误差]")
    print("  逐点 |d| (mm)      :", np.round(per_point, 4))
    print("  RMSE               : %.4f mm" % rmse)
    print("  MAE                : %.4f mm" % mae)
    print("  判定               :", "结果超差" if np.abs(err).max() > 1 else "合格")


def run_files(robot_path, base_path, robot_type):
    np.set_printoptions(precision=6, suppress=True)
    with open(robot_path, "r", encoding="utf-8", errors="ignore") as f:
        rob = parse_robot_data(f.read(), robot_type.upper())
    with open(base_path, "r", encoding="utf-8", errors="ignore") as f:
        base = parse_base_data(f.read())

    if len(rob) == 0 or len(base) == 0:
        sys.exit("未能从文件中解析出点位，请检查文件格式与机器人类型是否匹配")
    if len(rob) != len(base):
        sys.exit("选择的文件中点位不等：机器人 %d 点 / 三坐标 %d 点"
                 % (len(rob), len(base)))
    if len(rob) < 3:
        sys.exit("测量点至少为三点")

    t, euler, oat, quat, R, err, per_point, rmse, mae = calculate(rob, base)
    print("点位数:", len(rob))
    print("\nX Y Z (mm)\n", t)
    print("\nRx Ry Rz (°):", np.round(euler, 6))
    print("O  A  T  (°):", np.round(oat, 6))
    print("q1 q2 q3 q4 :", np.round(quat, 6))
    print("\n旋转矩阵 R\n", R)
    print("\nErr (dX dY dZ / mm)\n", err)
    print("\nRMSE %.4f mm | MAE %.4f mm" % (rmse, mae))
    if np.abs(err).max() > 1:
        print("结果超差")
    else:
        print("合格")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_files(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        self_test()
