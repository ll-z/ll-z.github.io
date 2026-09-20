# -*- coding: utf-8 -*-
"""
cli_demo.py —— 命令行自测 / 演示

用法：
    python cli_demo.py                          # 内置真值自测（含 TCP 自测）
    python cli_demo.py robot.dat base.txt KUKA  # 计算真实数据（单一刚体变换模型）
    python cli_demo.py robot.ls  base.txt FANUC
    python cli_demo.py robot.mod base.txt ABB

    python cli_demo.py robot.dat base.txt KUKA --tcp
        # 电极帽 TCP 自动标定：E6POS 记录的是编程 TCP，三坐标测的是电极帽实际触点，
        # 两者在工具坐标系下相差固定偏移 c，用 9 参数模型一起解出（与 WcsCal 同模型）。
"""
import math
import sys

import numpy as np

from ParameterCalculate import (
    calculate, calculate_with_tcp, orientation_spread, parse_base_data,
    parse_robot_data, parse_robot_data_with_pose,
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

    t, euler, abc, zyz, quat, R, err, per_point, rmse, mae = calculate(rob, base)

    print("\n[恢复精度]")
    print("  R 误差 (Frobenius) :", np.linalg.norm(R - R_true))
    print("  t 真值             :", t_true)
    print("  t 估计             :", np.round(t, 4))
    print("\n[输出参数]")
    print("  欧拉角 Rx Ry Rz (°):", np.round(euler, 4))
    print("  KUKA A  B   C  (°) :", np.round(abc, 4))
    print("  ZYZ 欧拉角     (°) :", np.round(zyz, 4))
    print("  四元数 (w,x,y,z)   :", np.round(quat, 6))
    print("\n[误差]")
    print("  逐点 |d| (mm)      :", np.round(per_point, 4))
    print("  RMSE               : %.4f mm" % rmse)
    print("  MAE                : %.4f mm" % mae)
    print("  判定               :", "结果超差" if np.abs(err).max() > 1 else "合格")
    tcp_self_test()


def tcp_self_test():
    """TCP 自动标定自测：构造已知 R、t、c 与「有变化」的工具姿态，验证能否恢复。"""
    print("\n" + "=" * 64)
    print("TCP 自测：9 参数模型（未知电极帽偏移 c + 变化的工具姿态）")
    print("=" * 64)
    R_true = euler_to_R(3.0, -5.0, 20.0)
    t_true = np.array([120.0, -85.0, 430.0])
    c_true = np.array([-80.0, 120.0, 300.0])
    rng = np.random.default_rng(7)
    base = rng.uniform(-300, 300, size=(12, 3)).round(3)
    Rt = np.array([euler_to_R(*a) for a in rng.uniform(-40, 40, size=(12, 3))])
    rob = ((R_true @ base.T).T + t_true) - (Rt @ c_true)     # E6POS = 实际触点 - Rtool·c
    rob += rng.normal(0, 0.02, base.shape)                   # 0.02mm 噪声

    t, euler, abc, zyz, quat, R, err, per_point, rmse, mae, tcp = calculate_with_tcp(rob, base, Rt)
    print("  工具姿态跨度        : %.2f°" % orientation_spread(Rt))
    print("  c 真值              :", c_true)
    print("  c 估计              :", np.round(tcp, 4),
          " (误差 %.4f mm)" % np.linalg.norm(tcp - c_true))
    print("  R 误差 (Frobenius)  : %.3e" % np.linalg.norm(R - R_true))
    print("  t 误差 (mm)         : %.4f" % np.linalg.norm(t - t_true))
    print("  KUKA A B C          :", np.round(abc, 4))
    print("  RMSE                : %.4f mm" % rmse)


def run_files(robot_path, base_path, robot_type, use_tcp=False):
    np.set_printoptions(precision=6, suppress=True)
    with open(robot_path, "r", encoding="utf-8", errors="ignore") as f:
        robot_text = f.read()
    with open(base_path, "r", encoding="utf-8", errors="ignore") as f:
        base = parse_base_data(f.read())

    Rt = None
    if use_tcp:
        rob, Rt = parse_robot_data_with_pose(robot_text, robot_type.upper())
        if Rt is not None:
            spread = orientation_spread(Rt)
            if spread < 5.0:
                sys.exit("各示教点姿态几乎相同（工具 Z 轴最大夹角仅 %.1f°），TCP 不可辨识；"
                         "请使用姿态有明显变化的示教点" % spread)
        else:
            sys.exit("这份机器人文件里没有解析到姿态（A/B/C / W/P/R / 四元数），"
                     "无法自动标定 TCP")
    else:
        rob = parse_robot_data(robot_text, robot_type.upper())

    if len(rob) == 0 or len(base) == 0:
        sys.exit("未能从文件中解析出点位，请检查文件格式与机器人类型是否匹配")
    if len(rob) != len(base):
        sys.exit("选择的文件中点位不等：机器人 %d 点 / 三坐标 %d 点"
                 % (len(rob), len(base)))
    if len(rob) < 3:
        sys.exit("测量点至少为三点")

    tcp = None
    if use_tcp:
        t, euler, abc, zyz, quat, R, err, per_point, rmse, mae, tcp = calculate_with_tcp(rob, base, Rt)
    else:
        t, euler, abc, zyz, quat, R, err, per_point, rmse, mae = calculate(rob, base)

    print("点位数:", len(rob))
    print("\nX Y Z (mm)\n", t)
    print("\nRx Ry Rz (°):", np.round(euler, 6))
    print("KUKA A B C (°):", np.round(abc, 6), "  (内旋 Z-Y'-X''，可直接填入 KUKA)")
    print("ZYZ 欧拉角 (°):", np.round(zyz, 6), "  (与 WcsCal 的 ZYZ 一组对应)")
    print("q1 q2 q3 q4 :", np.round(quat, 6))
    print("\n旋转矩阵 R\n", R)
    if tcp is not None:
        print("\n电极帽 TCP 偏移 c（工具坐标系, mm）\n", tcp)
        print("  |c| = %.6f mm  |  工具姿态跨度 %.3f°"
              % (np.linalg.norm(tcp), orientation_spread(Rt)))
    print("\nErr (dX dY dZ / mm)\n", err)
    print("\nRMSE %.4f mm | MAE %.4f mm" % (rmse, mae))
    print("结果超差" if np.abs(err).max() > 1 else "合格")


if __name__ == "__main__":
    use_tcp = "--tcp" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--tcp"]
    if len(argv) == 3:
        run_files(argv[0], argv[1], argv[2], use_tcp)
    else:
        self_test()
