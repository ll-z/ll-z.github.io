# 三维刚体配准与机器人 Base 标定 · 理论手册

> 配套实现：`index.html`（页面内嵌 Python/NumPy）、`python/ParameterCalculate.py`（命令行算法库）
> 本文整理本工具用到的全部数学、约定与工程判据，可作为校核依据与交接文档。

## 目录

- [0. 符号与约定速查](#0-符号与约定速查)
- [1. 坐标系与位姿](#1-坐标系与位姿)
- [2. 旋转的表示与转换](#2-旋转的表示与转换)
- [3. 刚体配准（Kabsch / 正交 Procrustes）](#3-刚体配准kabsch--正交-procrustes)
- [4. 带未知 TCP 的 9 参数模型](#4-带未知-tcp-的-9-参数模型)
- [5. 误差分析与质量判定](#5-误差分析与质量判定)
- [6. 机器人姿态约定与商业软件对应关系](#6-机器人姿态约定与商业软件对应关系)
- [7. 数值实践清单](#7-数值实践清单)
- [8. 参考文献](#8-参考文献)

---

## 0. 符号与约定速查

| 符号 | 含义 |
|---|---|
| $\mathbf{p} = (x,y,z)^{\mathsf T}$ | 三维点，**列向量** |
| $R \in SO(3)$ | $3\times3$ 旋转矩阵，$R^{\mathsf T}R=I$，$\det R = +1$ |
| $t \in \mathbb{R}^3$ | 平移向量 |
| $\mathbf{b}_i,\ \mathbf{q}_i$ | 第 $i$ 个点在**基坐标系 / 机器人坐标系**下的坐标 |
| $R_{\text{tool},i}$ | 第 $i$ 个示教点的**工具坐标系**姿态（由 A/B/C 或四元数还原） |
| $c$ | 电极帽 TCP 相对"编程 TCP"的固定偏移（工具坐标系下表达） |
| $\hat{\omega}$ | 向量 $\omega$ 的反对称矩阵（叉乘算子） |
| $\bar{a}$ | 点集的质心（均值） |

**核心变换方向（全工具统一）**：$\boxed{\ \text{机器人点} \approx R\cdot \text{基坐标系点} + t\ }$，即"基坐标系 → 机器人坐标系"。

**角度单位**：内部计算全用弧度，接口输出全用**度**。
**四元数顺序**：$(w,x,y,z)$，实部在前。
**欧拉角顺序**：见 [§2.2](#22-欧拉角三套约定与本工具的对应)（这里是历史坑点最多的部分）。

---

## 1. 坐标系与位姿

### 1.1 本工具涉及的坐标系

| 坐标系 | 别称 | 典型来源 | 在本工具中的角色 |
|---|---|---|---|
| **基坐标系 / 基准坐标系** | CMM 坐标系、工装坐标系、`bas` | 三坐标测量机测量点（`P1,x,y,z`） | 配准的**源**点集 $\mathbf{b}_i$ |
| **机器人世界/基座坐标系** | `$WORLD`、`$BASE`、`rob` | 示教器读数、`.dat`/`.ls`/`.mod` 里的 X/Y/Z | 配准的**目标**点集 $\mathbf{q}_i$ |
| **法兰坐标系** | flange、A6 | 由机器人运动学给出 | 一般不可见；工具偏移的参考 |
| **工具坐标系（TCP 框架）** | tool、UT | `.dat` 里的 A/B/C 姿态 | 用 $R_{\text{tool},i}$ 把 $c$ 转到世界系 |
| **编程 TCP** | 控制器里 `$TOOL` 的零点 | `.dat` 里的 X/Y/Z | 若与真实电极帽有偏差，该偏差就是 $c$ |
| **电极帽触点** | 实际测量点、焊枪电极 | 现场实际触碰点 | 三坐标测的"同一个点" |

> 关键认识：**三坐标测的是电极帽实际触点，而 `.dat` 里记录的 X/Y/Z 是控制器里"编程 TCP"的位置**。
> 二者不一定是同一个物理点——这就是是否需要在模型里引入 $c$ 的分界。

### 1.2 位姿与齐次变换

一个刚体位姿 = 旋转 + 平移：

$$
\mathbf{p}_{\text{机器}} = R\,\mathbf{p}_{\text{基}} + t
\qquad\Longleftrightarrow\qquad
\begin{bmatrix}\mathbf{p}\\1\end{bmatrix}_{\text{机器}}
= \underbrace{\begin{bmatrix}R & t\\ \mathbf{0}^{\mathsf T} & 1\end{bmatrix}}_{T\ \in\ SE(3)}
\begin{bmatrix}\mathbf{p}\\1\end{bmatrix}_{\text{基}}
$$

$T$ 的逆为：

$$
T^{-1} = \begin{bmatrix} R^{\mathsf T} & -R^{\mathsf T}t\\ \mathbf{0}^{\mathsf T} & 1\end{bmatrix}
$$

本工具输出的"**变换 ②（逆变换）**"就是它：$R_{\text{inv}} = R^{\mathsf T}$，$t_{\text{inv}} = -R^{\mathsf T}t$。

### 1.3 主动 / 被动、左乘 / 右乘

- 本文所有 $R$ 都是**主动旋转**（旋转点本身，坐标系不动）——工业机器人习惯如此。
- 若换成"被动"（坐标系的姿态描述），矩阵要取转置；不同品牌/软件的报告里混用，
  这是"看起来对了但实际上拧了 180°"的常见来源。
- 复合顺序：$A$ 先、$B$ 后 ⇒ $R = R_B R_A$（左乘新变换）。

---

## 2. 旋转的表示与转换

### 2.1 旋转矩阵与 SO(3)

$$
SO(3) = \{\,R \in \mathbb{R}^{3\times3} \mid R^{\mathsf T}R = I,\ \det R = +1\,\}
$$

三个性质在本工具中被反复使用：

1. **列（行）正交单位** ⇒ $\|R\mathbf{v}\| = \|\mathbf{v}\|$，距离与角度不变；
2. **$\det R = +1$** ⇒ 排除镜像（反射矩阵 $\det = -1$ 也会让球面配准"看起来对"，但物理上不可能）；
3. **3 个自由度**（9 个元素 − 6 个正交约束）。

### 2.2 欧拉角：三套约定与本工具的对应

同一份 $R$ 可以写成多套欧拉角，**数值完全不同**。本工具输出三套，含义如下（均已实测反解可还原 $R$，误差 $\sim10^{-16}$）：

| 本工具字段 | 契约（旋转顺序） | 常见别名 | 典型用途 |
|---|---|---|---|
| `euler` = `[Rx,Ry,Rz]` | $R = R_z(\mathrm{Rz})\,R_y(\mathrm{Ry})\,R_x(\mathrm{Rx})$ | 固定轴 X-Y-Z / "FIX XYZ" | 与 WcsCal 的 `FIX XYZ` 一组对应 |
| `abc` = `[A,B,C]` | $R = R_z(A)\,R_y(B)\,R_x(C)$ | **KUKA A/B/C**、内旋 Z-Y′-X″、WcsCal 的 `ZYX` | **可直接填进 KUKA 控制器** |
| `zyz` = `[a,b,c]` | $R = R_z(a)\,R_y(b)\,R_z(c)$ | ZYZ 欧拉角、WcsCal 的 `ZYZ` | 交叉核对 / 老脚本兼容 |

三者关系：**固定轴 XYZ 就是 KUKA ABC 的反序**，即 `euler = [C, B, A]`。

反解公式（`rotation_matrix_to_euler`，`rotation_matrix_to_zyz`）：

$$
\begin{aligned}
\text{KUKA ABC:}\quad
&A = \operatorname{atan2}(R_{10},\,R_{00})\\
&B = \operatorname{atan2}\!\big(-R_{20},\,\sqrt{R_{21}^2+R_{22}^2}\big)\\
&C = \operatorname{atan2}(R_{21},\,R_{22})
\end{aligned}
\qquad\qquad
\begin{aligned}
\text{ZYZ:}\quad
&a = \operatorname{atan2}(R_{12},\,R_{02})\\
&b = \operatorname{atan2}\!\big(\sqrt{R_{20}^2+R_{21}^2},\,R_{22}\big)\\
&c = \operatorname{atan2}(R_{21},\,-R_{20})
\end{aligned}
$$

**万向锁（gimbal lock）**：KUKA ABC 在 $B = \pm 90^\circ$（$\cos B \to 0$）时 $A$ 与 $C$ 只出现在 $A \pm C$ 的组合里，
本工具令 $A = 0$，取 $C = \operatorname{atan2}(-R_{20}R_{01},\,R_{11})$；
ZYZ 在 $b = 0^\circ/180^\circ$ 时同理令 $a = 0$。判据用 $\sqrt{R_{21}^2+R_{22}^2} < 10^{-12}$（ABC）/ $\sqrt{R_{20}^2+R_{21}^2} < 10^{-12}$（ZYZ）。

> ⚠️ **历史坑（已修正）**：本工具早期版本把 ZYZ 三元组标成了 "KUKA O A T"，
> 若直接当 A/B/C 填进 KUKA，姿态是错的（实测 $R_z(O)R_y(A)R_x(T)$ 与真实 $R$ 的矩阵元素最大差可达 2.0）。
> 现在 `abc` 字段是真 KUKA ABC，ZYZ 单独给出并明确标注。

### 2.3 四元数

$$
q = (w,x,y,z),\quad \|q\| = 1,\qquad
R(q) = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

关键性质：

- **双覆盖**：$q$ 与 $-q$ 表示同一旋转。比较四元数前必须先统一符号（例如规定 $w \ge 0$），否则"差一个负号"不是误差；
- **反解用 Shepperd 方法**：比较 $R_{00}+R_{11}+R_{22}$、$R_{00}-R_{11}-R_{22}$、$R_{11}-R_{00}-R_{22}$、$R_{22}-R_{00}-R_{11}$
  四个量的最大值，选对应分量开根做除法，避免除以接近 0 的数（本工具 `quaternion_calculator`）；
- 本工具的四元数与 SciPy `Rotation.as_quat()` 结果完全一致（已对照）。

### 2.4 旋转向量与指数映射

$$
R = \exp(\hat{\omega}) = I + \frac{\sin\theta}{\theta}\hat{\omega} + \frac{1-\cos\theta}{\theta^2}\hat{\omega}^2,
\qquad \theta = \|\omega\|
$$

其中 $\hat{\omega}$ 为反对称矩阵。作用：**在 SO(3) 上用无约束的 3 维向量做迭代**——
每步更新 $R \leftarrow R\exp(\hat{\delta\omega})$，结果天然正交，无需再投影回 $SO(3)$（见 [§4.4](#44-高斯-牛顿--levenberg-marquardt)）。

### 2.5 表示对照与选型

| 表示 | 参数 | 冗余 | 奇异性 | 插值 | 本工具用途 |
|---|---|---|---|---|---|
| 旋转矩阵 | 9 | 6 约束 | 无 | 差 | 求解、输出、可视化 |
| 欧拉角 | 3 | 无 | **有**（万向锁） | 差 | 给控制器/报告 |
| 四元数 | 4 | 1 约束 | 无 | **好** | 输出、ABB 输入 |
| 旋转向量 | 3 | 无 | $\theta=2\pi$ | 一般 | 迭代优化（LM） |

---

## 3. 刚体配准（Kabsch / 正交 Procrustes）

### 3.1 问题描述

给两两对应的点集 $\{\mathbf{b}_i\}_{i=1}^n$（基坐标系）与 $\{\mathbf{q}_i\}_{i=1}^n$（机器人坐标系），求

$$
(R^\star, t^\star) = \arg\min_{R\in SO(3),\ t\in\mathbb{R}^3} \sum_{i=1}^n \big\|\,R\,\mathbf{b}_i + t - \mathbf{q}_i\,\big\|^2
$$

**前提假设**（写在这里以免误用）：

- 两组点**一一对应**且顺序正确（本工具提供"按点号/按顺序"两种对齐方式）；
- 噪声主要来自测量，且近似各向同性、同方差、零均值；
- **不包含**工具的未知偏移——那是 [§4](#4-带未知-tcp-的-9-参数模型) 的内容。

### 3.2 消去平移：质心对齐

令 $\bar{\mathbf b} = \frac1n\sum\mathbf b_i$，$\bar{\mathbf q} = \frac1n\sum\mathbf q_i$，$\tilde{\mathbf b}_i = \mathbf b_i - \bar{\mathbf b}$，$\tilde{\mathbf q}_i = \mathbf q_i - \bar{\mathbf q}$。目标函数对 $t$ 是凸二次的，令其偏导为零：

$$
\frac{\partial}{\partial t}\sum\big\|R\mathbf b_i + t - \mathbf q_i\big\|^2 = 0
\ \Longrightarrow\
t = \bar{\mathbf q} - R\,\bar{\mathbf b}
$$

代回后问题只剩旋转（且与 $t$ 解耦）：

$$
R^\star = \arg\min_{R\in SO(3)}\sum_i\big\|R\tilde{\mathbf b}_i - \tilde{\mathbf q}_i\big\|^2
= \arg\max_{R\in SO(3)} \operatorname{tr}\!\big(R^{\mathsf T}H\big),
\qquad
H = \sum_i \tilde{\mathbf b}_i\,\tilde{\mathbf q}_i^{\mathsf T}
$$

（最后一步用了 $\|Ra-b\|^2 = \|a\|^2+\|b\|^2-2\,a^{\mathsf T}R^{\mathsf T}b$ 与迹的循环性质。）

### 3.3 SVD 闭式解

对交叉协方差矩阵做奇异值分解 $H = U\Sigma V^{\mathsf T}$，则

$$
\boxed{\ R^\star = V\,\operatorname{diag}(1,1,\det(VU^{\mathsf T}))\,U^{\mathsf T}\ }
$$

其中 $\operatorname{diag}(1,1,d)$ 的第三个对角元 $d=\det(VU^{\mathsf T})$ 用来**修正反射**（见 §3.4）。
实现上等价于：$R \leftarrow V^{\mathsf T}U^{\mathsf T}$… 注意本仓库代码写的是

```python
H  = S.T @ T            # S = source - mean, T = target - mean   -> H = Σ (b-b̄)(q-q̄)ᵀ
U, s, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:      # 反射修正
    Vt[2, :] *= -1
    R = Vt.T @ U.T
t = mu_t - R @ mu_s
```

与上式的对应：$U_{\text{numpy}} = U$，$V_{\text{numpy}} = V^{\mathsf T}$。翻转 $V$ 的最后一行等价于 $V\operatorname{diag}(1,1,-1)$，
即 $\det$ 修正的标准做法。

### 3.4 为什么要做反射修正

无约束的正交矩阵解允许 $\det R = -1$（镜像）。若数据存在退化或噪声较大，最优正交解可能是反射矩阵，
这时强行使用会得到"物理上不可能"的变换（右手系变左手系）。
$SO(3)$ 上的最优解就是把 $\Sigma$ 的最后一个（最小）奇异值取反，代价仅增加 $\sigma_3$。

### 3.5 统计解释

若 $\mathbf q_i = R\mathbf b_i + t + \varepsilon_i$，$\varepsilon_i \sim \mathcal N(0,\sigma^2 I)$ 独立同分布，
则最小二乘解 $=$ 极大似然解。更一般的噪声（各向异性 $\Sigma$）应改用加权最小二乘：

$$
\min \sum_i (\dots)^{\mathsf T}\Sigma^{-1}(\dots)
$$

本工具默认等权；**如果某个点测量精度明显不同，应改用剔除或加权**（页面支持逐点勾选，等价于 0/1 权重）。

### 3.6 退化与病态

| 情况 | 后果 |
|---|---|
| $n = 1$ | 旋转完全不可观测（只能求平移） |
| $n = 2$ | 只能确定一个轴向，绕该轴任意旋转都最小 |
| $n = 3$ 且三点共线 | 同 $n=2$ |
| 点集近似共面 | $H$ 的最小奇异值与次小奇异值接近，$R$ 对噪声敏感（**"扁平"点集最不受控**） |
| 点集几何分布均匀（无共线、跨越三个方向） | 条件数好，推荐 |

工程建议：示教点尽量**分散**（点间距 $\ge$ 数十毫米、避免全部近似共面），这对 [§4](#4-带未知-tcp-的-9-参数模型) 同样重要。

### 3.7 不确定度估计

残差平方和 $S = \sum\|\hat{\mathbf e}_i\|^2$，未知参数 $p = 6$（$R$ 3 + $t$ 3），观测数 $3n$，则噪声方差的无偏估计

$$
\hat\sigma^2 = \frac{S}{3n - p}
$$

参数协方差可用数值雅可比 $J$ 近似 $\widehat{\mathrm{cov}}(\theta) \approx \hat\sigma^2 (J^{\mathsf T}J)^{-1}$。
$n$ 小时（如本工具的 4 点示例，$3n-p = 6$）不确定度的估计本身很粗糙，**不要把 RMSE 当成精度承诺**。

---

## 4. 带未知 TCP 的 9 参数模型

### 4.1 物理背景

现场标定流程：把示教点"对齐"到夹具上由三坐标测出的基准点，机器人示教器记录每个点的 X/Y/Z（控制器里**当前生效工具**的 TCP 位置）和 A/B/C（该点的工具姿态）。

问题在于：**示教器记录的 X/Y/Z 是"编程 TCP"，未必是电极帽实际触点**。可能造成偏差的原因：

- 控制器里 `$TOOL` 为空/是旧值/被换过；
- 电极帽磨损、更换、修磨后长度变化；
- 焊枪伺服轴（如 KUKA 的 E1）实际行程与模型不一致。

两者之间是一个**在工具坐标系下恒定**的偏移 $c$（换点不换工具）：

$$
\text{实际触点} = \mathbf q_i + R_{\text{tool},i}\,c
$$

> 注意：这里的 $c$ 是"实际触点 − 编程 TCP"在工具坐标系下的表达，**不是**"从法兰量起的完整 TCP"。
> 商业软件（如 LEADOPTICS WcsCal）输出的 `Calculated TCP` 与它是同一个量（实测 9 个现场案例中最大差 0.64 mm）。

### 4.2 数学模型

把触点代入刚体配准模型，得到 **9 参数模型**（$R$ 3 + $t$ 3 + $c$ 3）：

$$
\boxed{\ R\,\mathbf b_i + t \;=\; \mathbf q_i + R_{\text{tool},i}\,c \qquad (i = 1\dots n)\ }
$$

残差定义：

$$
\mathbf e_i = R\mathbf b_i + t - \mathbf q_i - R_{\text{tool},i}\,c
$$

**结构性质**（决定了求解策略）：

- 对 $(R,t,c)$ 是**线性**的（把 $R$ 的 9 个元素当成自由变量时）；
- 约束 $R \in SO(3)$ 使整体**非凸**；
- 对 $c$ 严格线性、对 $R$ 是非凸旋转约束 —— 这是后续所有算法设计的出发点。

### 4.3 为什么"交替最小化"会失败

最自然的做法是块坐标下降（BCD）：

```
重复：
  ① 固定 c，用 Kabsch 求 (R, t)        # 对 (R,t) 全局最优
  ② 固定 (R,t)，最小二乘求 c           # 对 c 全局最优（线性问题）
```

每一步都是子问题的**全局最优**，但**块最优点 ≠ 最优点**。本工具在一批 9 个现场案例上实测：

| 方法 | 平均 RMSE（3D 范数 RMS） |
|---|---|
| 交替最小化（40 个随机起点） | 0.487 mm |
| 多起点 Gauss-Newton/LM | **0.429 mm** |

个别案例差异极大（如某站 0.71 → 0.41 mm）。原因：目标函数在 $R$ 方向上非凸，
BCD 会稳定在**鞍点**（对每个坐标块都是极小，但不是联合极小）。
**结论：不要用交替最小化求解本模型**（这也是"为什么商业软件与自研脚本结果差几十毫米"的常见根因之一）。

### 4.4 高斯-牛顿 / Levenberg-Marquardt

把旋转用局部旋转向量参数化（右乘扰动）：

$$
R \leftarrow R\,\exp(\hat{\delta\omega}),
\qquad
c \leftarrow c + \delta c,
\qquad
t \leftarrow t + \delta t
$$

对残差求一阶展开（$\mathbf b$ 为该点在基坐标系下的坐标）：

$$
\delta \mathbf e_i = -R\,[\mathbf b_i]_\times\,\delta\omega \;-\; R_{\text{tool},i}\,\delta c \;+\; \delta t
$$

于是雅可比 $J \in \mathbb{R}^{3n\times 9}$ 的第 $i$ 块行为

$$
J_i = \big[\underbrace{-R\,[\mathbf b_i]_\times}_{3\times3}\ \ \underbrace{-R_{\text{tool},i}}_{3\times3}\ \ \underbrace{I_3}_{3\times3}\big]
$$

（列分组依次是 $\delta\omega,\ \delta c,\ \delta t$。）

**LM 迭代**：

$$
(J^{\mathsf T}J + \lambda\,\mathrm{diag}(J^{\mathsf T}J))\,\delta = -J^{\mathsf T}\mathbf e
$$

- 新点残差下降 ⇒ 接受并**减小** $\lambda$（本工具 $\lambda \leftarrow \lambda/3$）；
- 残差上升 ⇒ 拒绝并**增大** $\lambda$（$\lambda \leftarrow 10\lambda$），重解；
- 收敛判据：$\|\delta\| < 10^{-14}$ 或达到迭代上限。

伪代码（`rigid_transform_with_tcp` / `_lm_tcp`）：

```python
for c0 in 多起点:
    R0, t0 = kabsch(base, rob + Rtool @ c0)     # 用 Kabsch 给出好的初值
    R, t, c = LM(base, rob, Rtool, R0, t0, c0)  # 9 参数迭代
    cost    = || R@base + t - (rob + Rtool@c) ||
    保留 cost 最小的一组
```

### 4.5 多起点与全局性

由于非凸，单起点可能落到局部极小。本工具采用**确定性多起点**（可复现，不依赖随机数）：

$$
c_0 \in \{\mathbf 0\} \cup \{\,r\cdot\mathbf u_k \,\big|\; r \in \{0.5s,\, s\},\ \mathbf u_k = \text{Fibonacci 球面 24 个方向}\,\}
$$

其中 $s$ = 机器人点云的平均半径（尺度自适应），共 $1 + 2\times24 = 49$ 个起点。

实测（9 个现场案例）：**49 个确定性起点的结果与 200 个随机起点完全一致（RMSE 差 0.000000 mm）**，
可认为已稳定找到全局最优。

> 若点数很多（>100）且追求速度，可减少方向数（`ndir`），但会牺牲鲁棒性。

### 4.6 可辨识性（什么时候 $c$ 求不出来）

把方程整理成"未知量 $\mathbf x = (c,\ t)$"的形式：

$$
R_{\text{tool},i}\,c - t \;=\; R\mathbf b_i - \mathbf q_i
$$

若所有点的工具姿态相同（$R_{\text{tool},i} \equiv R_{\text{tool}}$），则

- $c$ 与 $t$ 只以组合 $R_{\text{tool}}c - t$ 出现 —— **只有 3 个方程未知量、但有 6 个未知量**；
- 任意 $c$ 都能通过 $t' = t - R_{\text{tool}}c + R_{\text{tool}}c^\ast$ 补偿 ⇒ **$c$ 不可辨识**。

所以需要**姿态变化**。本工具用工程判据：

$$
\text{工具 Z 轴两两夹角的最大值} \ge 5^\circ
$$

否则拒绝计算并提示。反之，姿态跨度越大，$c$ 的估计越稳定（推荐 $\ge 20^\circ$）。
另外一个必要条件：$\{R_{\text{tool},i}\}$ 不能只绕单轴旋转（那样沿该轴的偏移分量仍不可辨识）。

### 4.7 与手眼标定（hand–eye）的关系

本模型属于 **robot–world / hand–eye 标定（AX = ZB）** 家族：把 $R_{\text{tool},i}$ 看成"工具在机器人世界系中的姿态"，
$c$ 看成"工具坐标系中的固定点"，则问题等价于求世界系与外部测量系之间的刚体变换 + 一个固定点。
常见解法有 Tsai–Lenz、Park–Martin、Daniilidis 等**闭式解法**（基于旋转轴/对偶四元数），
本工具选择**直接非线性最小二乘**的原因：

- 闭式解法对噪声敏感、且通常要求姿态变化足够大（同样是可辨识性条件）；
- 数值 LS 可直接处理工程中的缺失点、粗差点（逐点勾选）与加权；
- 只要多起点足够，可获得比闭式解法更小的残差（本工具在 9 个现场案例上 RMSE 0.429 mm，优于 WcsCal 报告的 Ravg 0.413 mm；两者定义差异见 §5.2）。

### 4.8 自由度与最少点数

| 模型 | 未知参数 $p$ | 观测方程 | 残差自由度 |
|---|---|---|---|
| 单一刚体变换 | 6 | $3n$ | $3n - 6$ |
| 9 参数（含 $c$） | 9 | $3n$ | $3n - 9$ |

- 必要（但不充分）条件：$n \ge 2$（刚体）/$n \ge 3$（含 $c$）；
- 工程建议：**至少 6～8 点，且空间分布分散、姿态跨越大**；
- 点数小于参数数时，问题欠定，求解器可能给出"拟合完美但物理荒谬"的解（例如 $c$ 跑到几百米外）。

---

## 5. 误差分析与质量判定

### 5.1 残差及其坐标系 —— 一个常被忽略的坑

本工具（与早期脚本一致）定义残差为

$$
\mathbf e_i = \text{目标} - \text{预测} = \big(\mathbf q_i + R_{\text{tool},i}c\big) - \big(R\mathbf b_i + t\big)
$$

即**在机器人坐标系下表达**。另一些实现（以及部分报告）把残差表达在**基坐标系**、并且**取相反符号**：

$$
\mathbf e_i^{\text{(基)}} = -R^{\mathsf T}\mathbf e_i
$$

两种写法**逐点范数完全相同**（旋转不改变长度），但分量 $(\mathrm{d}X,\mathrm{d}Y,\mathrm{d}Z)$ 的数值不一样。
对比不同来源的报告时，先核对这一点（本工具与某参考报告实测：范数完全一致、分量差 0.88 mm 级别）。

### 5.2 统计量

$$
\text{逐点误差}\quad d_i = \|\mathbf e_i\|_2,
\qquad
\text{RMSE} = \sqrt{\frac1n\sum_i d_i^2},
\qquad
\text{MAE} = \frac1n\sum_i d_i,
\qquad
\text{最大分量误差} = \max_{i,k}|e_{i,k}|
$$

- **RMSE** 对大误差更敏感（含平方），适合做"总体拟合优度"；
- **MAE** 更稳健，接近商业软件报告里的 `Ravg`（其 `Rmin/Ravg/Rmax` 就是 $d_i$ 的最小/平均/最大值）；
- **最大分量误差** 用于判定"是否超差"最保守（任何方向的单轴偏差都看得见）。

### 5.3 阈值判定

本工具默认阈值 **±1 mm**：

- 残差 $\le 1$ mm ⇒ 显示"合格"；
- 否则显示"结果超差"。

阈值可随工艺要求调整（例如点焊枪 TCP 复现性通常要求 0.5 mm 级，而大件抓取工装 1～2 mm 也可接受）。
判定只是**过滤器**，不能替代对点分布的检查。

### 5.4 异常点识别与剔除

现场常见异常点来源：示教时碰撞/未到位、三坐标测错点、点位编号写错、电极帽在测量中途更换。

处理流程（与商业软件一致）：

1. 先全部参与，解出 $R,t,c$；
2. 看逐点 $d_i$，把明显偏大的点（例如 $>3\times$MAE，或超过阈值）**取消勾选**；
3. 重新计算；
4. 重复 2–3 次直到稳定；
5. **至少保留 $3n - p$ 的自由度**（含 $c$ 的模型建议保留 $\ge 8$ 点），否则剔除会掩盖真实误差。

页面上的"点位配对与选择"面板即为此设计（全选/全不选/反选 + 逐点勾选）。

---

## 6. 机器人姿态约定与商业软件对应关系

### 6.1 KUKA（`.dat` / `E6POS`）

```
DECL E6POS XP1={X 301.597,Y -138.044,Z 627.122,A 12.34,B 5.67,C -8.9,S 2,T 3,E1 ...}
```

- X/Y/Z：TCP 位置（毫米，相对于 `$BASE`/`$WORLD`）；
- **A/B/C：内旋 Z-Y′-X″**，即

$$
R_{\text{tool}} = R_z(A)\,R_y(B)\,R_x(C)
$$

本工具 `rotation_from_abc(A,B,C)` 即此式；实测反解（`rotation_matrix_to_euler`）能原样还原输入 ABC，
并在 9 个现场案例中与商业软件的 `ZYX` 组完全对应。
- S/T：状态字/转弯区，本工具忽略；E1…E6：外部轴（焊钳伺服），本工具只取姿态用。

### 6.2 FANUC（`.ls`）

```
P[1]{ GP1: UF : 2, UT : 1, X = 1715.906 mm, Y = 760.369 mm, Z = 1194.894 mm,
       W = -.901 deg, P = 2.854 deg, R = -89.751 deg   };
```

- W/P/R 为**固定轴 X-Y-Z**（绕世界 X、再绕世界 Y、再绕世界 Z），故

$$
R_{\text{tool}} = R_z(R)\,R_y(P)\,R_x(W)
$$

本工具 `rotation_from_abc(R, P, W)` 即此式（注意参数顺序被换掉了）。
- UF/UT 分别是用户坐标系与工具号，说明这些点是在某个 UFRAME/UTOOL 下记录的；
  与本模型配合时，只要 X/Y/Z 与姿态同属一个坐标系即可。
- ⚠️ FANUC 分支尚未用现场数据验证（现有案例均为 KUKA），使用前建议先拿一组已知结果核对。

### 6.3 ABB（`.mod`）

```
MoveL [[301.597,-138.044,627.122],[0.12,0.68,0.71,0.10],[0,0,0,0],[9E9,...]], v200, fine, tool0;
```

- 第一个数组是位置；第二个数组是**四元数 $[q_1,q_2,q_3,q_4] = [w,x,y,z]$**；
  本工具 `rotation_from_quat` 即 §2.3 的公式（内部再归一化）。
- ⚠️ 同样尚未用现场数据验证。

### 6.4 与 LEADOPTICS WcsCal 输出的对应（9 个现场案例实测）

| WcsCal 输出 | 本工具对应 | 实测偏差 |
|---|---|---|
| `Robot Locations` | `.dat` 里的 X/Y/Z 原值 | 完全相同（0.000 mm） |
| `Base Location` | 三坐标文件同点号的行 | 完全相同 |
| `Orientation Matrix`（3×3） | $R^{\mathsf T}$（即"变换 ②/逆变换"的 $R$） | 元素最大差 $1.9\times10^{-4}$（≈0.011°） |
| `Robot----To----Base` 的平移 | $t$（变换 ① 的平移） | ≤ 0.62 mm |
| `Base----To----Robot` 的平移 | $-R^{\mathsf T}t$（变换 ② 的平移） | ≤ 0.62 mm |
| `ZYX, ...` | `abc`（KUKA A/B/C） | ≤ 0.01° |
| `FIX XYZ, ...` | `euler` | ≤ 0.01° |
| `ZYZ, ...` | `zyz` | ≤ 0.01° |
| `Quart, ...` | `quat`（注意双覆盖符号） | 符号可能相反 |
| `Calculated TCP` | `tcp`（即 $c$） | ≤ 0.64 mm |
| 逐点 `err_X, err_Y, err_Z`（`err_R` 为范数） | 逐点残差（**坐标系/符号可能不同**，见 §5.1） | 范数一致 |
| `Rmin, Ravg, Rmax` | 逐点误差的最小/平均/最大 | 本工具 RMSE 0.429 / MAE 0.399 vs 其 Ravg 0.413 |

**模型级验证（最强证据）**：用 WcsCal **自己报告**的 $R$、平移、TCP 回算逐点残差，
9 个案例中 8 个与其报告的 `err_X/err_Y/err_Z` 最大偏差 $< 1.3\times10^{-3}$ mm
（剩余 1 例 `BSL440ROB04` 偏差 0.78 mm，说明它报告的 $R/t/c$ 并非该模型的最小二乘精确解；
本工具对该例的解与其 `Ravg` 一致，RMSE 0.396 mm vs 其 Ravg 0.334 mm）。
这证明双方是**同一个数学模型**，差异只来自数值实现与解的正则性。

> 另一个真实数据坑（`BSL440ROB04`）：`.dat` 里**缺 `pos12`**（点位名从 `pos11` 直接跳到 `pos13`），
> 而三坐标文件里**有 `p12`**，两侧点号不再一一对应 —— 收费软件的列表是按**文件顺序**配对的
> （`pos13` ↔ `p12`、`pos14` ↔ `p13` …）。本工具默认的"按点号"对齐在这种数据上会错配（残差 293 mm），
> 切换成"按顺序"即与商软一致（RMSE 0.396 mm）。**遇到两侧点号体系不一致时，优先试"按顺序"并用配对表核对。**

### 6.5 其他常见坑

1. **欧拉角顺序**：同一组数按不同顺序解释，结果完全不同（见 §2.2）。
2. **固定轴 vs 内旋**：`Rz Ry Rx` 与 `Rx Ry Rz` 不是同一件事。
3. **四元数顺序/符号**：$(w,x,y,z)$ vs $(x,y,z,w)$；$q$ vs $-q$。
4. **角度与弧度**：KUKA 的 A/B/C 是度，公式里必须换成弧度。
5. **单位**：本工具内部一致使用毫米；若混合米/毫米，会在 $t$ 与 $c$ 上出现 1000 倍错误。
6. **残差坐标系**：见 §5.1。
7. **点号重复**：例如 `.dat` 里两个点都叫 `XLHP007`（真实案例），按点号对齐会把两个机器人点映射到同一个三坐标点。

---

## 7. 数值实践清单

**数据准备**

- [ ] 机器人文件与三坐标文件**点数一致**（不一致就按点号对齐，或在文件里删行）
- [ ] 点号唯一、两侧编号一致（`POS07` ↔ `P7`）
- [ ] 单位统一（mm）；姿态单位与解析约定一致
- [ ] 抽取结果条数 = 源文件条数（本工具会在解析后立即给出点数）

**可解性**

- [ ] 点数 $n \ge 8$（含 TCP 时），$n \ge 4$（单一刚体变换）
- [ ] 点集在三个方向上都有跨度，避免近似共线/共面
- [ ] 要算 TCP 时，工具姿态跨度 $\ge 5^\circ$（推荐 $\ge 20^\circ$），且不是绕单轴变化

**结果自检**

- [ ] 逐点误差是否有明显离群点；剔除后结论是否稳定
- [ ] 把解出的 $R,t,c$ 代回，逐点残差与报告一致
- [ ] 旋转矩阵正交性（$\|R^{\mathsf T}R - I\|_\infty < 10^{-12}$）、$\det R = +1$
- [ ] KUKA ABC 与固定轴 XYZ 是否互为反序（快速一致性检查）
- [ ] 与历史/商软结果对比时，先核对 §6.4 的对应关系再比数字

**使用**

- [ ] `.dat` 里 TCP 已是实际触点 ⇒ 选"机器人点 = 电极帽触点"，**不要**去标定 $c$
- [ ] 不确定 ⇒ 用"自动判断"：残差已在阈值内就不标定 TCP
- [ ] 需要标定 ⇒ 用"机器人点 = 编程 TCP"，并把输出的 $c$ 写回控制器 `$TOOL`（注意 $c$ 是"实际 − 编程"的增量）

---

## 8. 参考文献

**配准与旋转**

1. W. Kabsch, *A solution for the best rotation to relate two sets of vectors*, Acta Cryst. A32, 1976.
2. B. K. P. Horn, *Closed-form solution of absolute orientation using unit quaternions*, JOSA A4, 1987.
3. S. Umeyama, *Least-squares estimation of transformation parameters between two point patterns*, IEEE TPAMI 13(4), 1991.
4. K. S. Arun, T. S. Huang, S. D. Blostein, *Least-squares fitting of two 3-D point sets*, IEEE TPAMI 9(5), 1987.
5. S. W. Shepperd, *Quaternion from rotation matrix*, J. Guidance and Control 1(3), 1978.
6. J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, arXiv:1812.01537, 2018.

**手眼标定 / 机器人标定**

7. R. Y. Tsai, R. K. Lenz, *A new technique for fully autonomous and efficient 3D robotics hand/eye calibration*, IEEE T-RA 5(3), 1989.
8. F. C. Park, B. J. Martin, *Robot sensor calibration: solving AX = XB on the Euclidean group*, IEEE T-RA 10(5), 1994.
9. K. Daniilidis, *Hand-eye calibration using dual quaternions*, IJRR 18(3), 1999.
10. R. Horaud, F. Dornaika, *Hand-eye calibration*, IJRR 14(3), 1995.

**优化**

11. K. Levenberg, *A method for the solution of certain non-linear problems in least squares*, Quart. Appl. Math. 2, 1944.
12. D. W. Marquardt, *An algorithm for least-squares estimation of nonlinear parameters*, SIAM J. Appl. Math. 11(2), 1963.
13. J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006（第 10 章：非线性最小二乘）。

**教材与手册**

14. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., CUP, 2004（Procrustes / 正交配准）。
15. KUKA System Software 8.x 操作/编程手册（`$POS_ACT`、`E6POS`、A/B/C 与工具坐标系定义）。
16. FANUC Robot 操作说明书（`W/P/R` 姿态定义、`UF`/`UT`）。
17. ABB RAPID 参考手册（`robtarget`、四元数 `q1..q4`）。
18. LEADOPTICS *WcsCal Cell Alignment* 输出说明（本仓库对该输出做了实测对照，见 §6.4）。

---

> 本文中所有"实测"数据均来自本仓库同批现场案例（9 个机器人站，KUKA，WcsCal 对照）；
> 复现脚本思路：解析 `.dat`/`.txt` → 配对 → 求解 → 与结果文件逐项比对。