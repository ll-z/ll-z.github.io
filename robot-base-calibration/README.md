# 坐标变换计算 · 机器人 Base 标定

三维刚体配准工具：用三坐标测量机（CMM）测得一组基准点，机器人 TCP 示教到同样的物理点，通过 **SVD 最小二乘刚体变换（Kabsch 算法）** 求解机器人坐标系与基坐标系之间的旋转矩阵 R 和平移向量 t，输出欧拉角（XYZ）、KUKA OAT 角、四元数与逐点配准误差。

## 特性

- 🌐 **纯静态网页演示**：`index.html` 单文件，无需后端，可直接部署 GitHub Pages
- 🐍 **浏览器内运行真实 Python**：通过 Pyodide（WebAssembly）加载 NumPy，算法在页面中实际执行，可在页面底部查看完整源码
- 🤖 **三种机器人格式**：KUKA `.dat`（E6POS）/ FANUC `.ls` / ABB `.mod`（MoveL、MoveP）
- 📊 **误差可视化**：双坐标系 3D 点云对比（机器人坐标系 / 基坐标系叠加），误差热力着色 + 残差连线，逐点误差表与 RMSE / MAE 统计
- ✅ **超差判定**：阈值 ±1 mm，自动给出「合格 / 结果超差」
- 🔗 **点号对齐 + 勾选参与计算的点**：机器人程序按点名（`POS07` → `P7`）、三坐标文件按首列点号（`P7,…`）自动匹配，两侧顺序不一致也能对上；两侧点号体系不同（例如机器人 `XLHP005` 对三坐标 `P1`）时会自动退回「按文件顺序」并在提示里说明；配对表里可逐点勾选参与计算，用于剔除异常点后重算（与收费软件一致）
- 🧭 **点位关系可选**：页面上可直接标记「机器人点 = 电极帽触点」（单一刚体变换）还是「机器人点 = 编程 TCP」（自动标定偏移 `c`），也可交由自动判断（姿态有变化就标定）
- 🎯 **电极帽 TCP 自动标定**：现场 `.dat` 里的 `E6POS` 记录的是「编程 TCP」，而三坐标测的是电极帽实际触点，二者在工具坐标系下相差一个固定偏移 `c`。勾选后按 9 参数模型 `R·base + t = rob + R_tool·c`，用多起点 Gauss-Newton / LM 同时解出 `R`、`t`、`c`（与 WcsCal 等收费软件同一模型）
- 🔢 三坐标文件同时兼容「点号,X,Y,Z」与「X,Y,Z」两种列格式
- 💾 结果可导出为 txt

## 目录结构

```
├── index.html                  # 网页演示（单文件应用，GitHub Pages 入口）
├── .nojekyll                   # 关闭 Jekyll 处理
└── python/
    ├── ParameterCalculate.py   # 算法库（现代 NumPy，兼容 1.x / 2.x）
    └── cli_demo.py             # 命令行自测 / 真实数据计算
```

## 算法

变换约定：`rob ≈ R @ base + t`（列向量），`R` 正交且 `det(R) = +1`。

1. 求两点集质心并去质心：`S = source − μs`，`T = target − μt`
2. 互协方差矩阵：`H = Sᵀ @ T`
3. SVD 分解 `H = U·Σ·Vᵀ`，最优旋转 `R = V·Uᵀ`；若 `det(R) < 0` 则翻转 `V` 最后一行修正反射
4. 平移：`t = μt − R·μs`
5. 欧拉角（含万向锁退化处理）、四元数（Shepperd 方法 + 归一化）
6. 残差：`err = target − (R @ source + t)`，统计逐点范数 / RMSE / MAE

> 已移除旧版 `np.mat` / `np.tile` 用法（`np.mat` 自 NumPy 2.0 起被移除），
> 改用 `np.asarray`、`@` 运算符与广播，兼容 NumPy 1.x 与 2.x。

### 电极帽 TCP 自动标定（可选功能，页面默认关闭）

现场标定时，机器人程序中的 `E6POS` 是控制器里的**编程 TCP**，而三坐标测量的是**电极帽实际触点**，
两者在工具坐标系下相差一个固定偏移 `c`：

```
实际触点 = rob_i + R_tool(A,B,C)_i · c        # R_tool 由各点姿态 A/B/C 还原
配准关系 = R · base_i + t   = 实际触点
```

未知量 9 个（R 3 + t 3 + c 3）。页面上的「点位关系」三选一对应：

| 选项 | 含义 | 模型 |
|---|---|---|
| 自动判断 | 先用单一刚体变换试算：残差已在 ±1 mm 内就不标定 TCP；残差偏大**且**工具 Z 轴跨度 ≥ 5° 才启用 TCP 标定 | 自动 |
| 机器人点 = 电极帽触点 | `.dat` 里的点已经是电极帽实际触点（TCP 已标定好，**不需要算 TCP**） | `rob ≈ R·base + t` |
| 机器人点 = 编程 TCP | `.dat` 里是控制器里的编程 TCP，需要算偏移 `c` | `R·base + t = rob + R_tool·c` |

算法要点：

1. 目标函数**对 `c` 线性、对 `R` 非凸**；「给定 c 求 R,t → 给定 R,t 求 c」的交替最小化会**停在鞍点**
   （实测平均 0.487 mm vs 最优 0.429 mm），故采用**确定性多起点 + Gauss-Newton / Levenberg-Marquardt**：
   `c` 的起点为 0 加球面 24 方向 × 半径 0.5 / 1.0 倍点云尺度，共 49 个，取残差最小者；
   R 用指数映射 `R ← R·exp([δω])` 更新，天然保持正交。实测 49 个确定性起点与 200 个随机起点结果完全一致。
2. **可辨识性判断**：各点工具 Z 轴最大夹角 < 5° 时提示「TCP 不可辨识」
   （姿态没有变化时 `c` 与 `t` 耦合，解不唯一）。
3. **与收费软件（LEADOPTICS WcsCal）实测对比**（9 个现场案例）：旋转矩阵最大差 1.9e-4（≈0.011°）、
   平移最大差 0.62 mm、TCP 最大差 0.64 mm；本算法 RMSE 0.429 / MAE 0.399，略优于其 Ravg 0.413。
   另用其报告值回算逐点残差，8/9 例最大偏差 < 1.3e-3 mm（剩余 1 例的 `.dat` 与结果文件点位不一致，已排除）。

> 注：FANUC 的 `W/P/R` 按固定角 XYZ（`Rz(R)·Ry(P)·Rx(W)`）解析，ABB 按 `[q1,q2,q3,q4]=[w,x,y,z]` 四元数解析，
> 这两种格式尚未用现场数据验证过，使用前建议先与已知结果核对。

## 本地使用

### 网页演示

直接用浏览器打开 `index.html`（需联网加载 Pyodide CDN），或：

```bash
python -m http.server 8000
# 打开 http://localhost:8000
```

页面内置了三种机器人格式的示例数据，可一键载入；勾选「注入 2mm 超差点」可演示超差检出。

### Python 命令行

```bash
cd python
pip install numpy

# 内置真值自测
python cli_demo.py

# 计算真实数据
python cli_demo.py robot.dat base.txt KUKA
python cli_demo.py robot.ls  base.txt FANUC
python cli_demo.py robot.mod base.txt ABB

# 自动标定电极帽 TCP（9 参数模型，需文件含姿态 A/B/C 或四元数）
python cli_demo.py robot.dat base.txt KUKA --tcp
```

> 网页端还支持「点号对齐」和「逐点勾选」；命令行版按文件顺序一一对应，需要剔除点请先把对应行删掉。

不加 `--tcp` 时是单一刚体变换模型（`rob ≈ R·base + t`）；若现场 `.dat` 里的 TCP 与电极帽实际触点不一致，
该模型的残差会达到几十到几百毫米，此时必须加 `--tcp`。

## 部署到 GitHub Pages

### 方式一：个人主页仓库（`用户名.github.io`）

```bash
git init
git add .
git commit -m "机器人 Base 标定演示"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<你的用户名>.github.io.git
git push -u origin main
```

推送后访问 `https://<你的用户名>.github.io/` 即可（作为个人主页的根页面）。

### 方式二：项目页面

1. 新建任意仓库（如 `robot-base-calibration`），把本目录内容推送到 `main` 分支：
   ```bash
   git init && git add . && git commit -m "init"
   git remote add origin https://github.com/<你的用户名>/robot-base-calibration.git
   git push -u origin main
   ```
2. 仓库页面 → **Settings → Pages → Source** 选择 `Deploy from a branch`，分支选 `main`、目录选 `/ (root)`，保存
3. 约 1 分钟后访问 `https://<你的用户名>.github.io/robot-base-calibration/`

> 注意：页面通过 CDN 加载 Pyodide 与 NumPy，访问时需要联网。

## 输入数据格式

| 类型 | 格式示例 |
|------|----------|
| KUKA `.dat` | `DECL E6POS XP1={X 301.597,Y -138.044,Z 627.122,A ...,S 2,T 3}` |
| FANUC `.ls` | `X =  301.597 mm,  Y =  -138.044 mm,  Z =  627.122 mm,` |
| ABB `.mod` | `MoveL [[301.597,-138.044,627.122],[...],[...],[...]], v200, fine, tool0;` |
| 三坐标 `.txt`（带点号） | `P1,164.374,-36.673,215.159`（逗号分隔，允许表头行） |
| 三坐标 `.txt`（无点号） | `5782.200550,-346.425733,1770.947319`（每行 X,Y,Z，现场三坐标常见导出格式） |

机器人点数与三坐标点数必须一致且 ≥ 3 点，点与点一一对应。

## License

MIT
