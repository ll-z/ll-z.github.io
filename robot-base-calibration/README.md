# 坐标变换计算 · 机器人 Base 标定

三维刚体配准工具：用三坐标测量机（CMM）测得一组基准点，机器人 TCP 示教到同样的物理点，通过 **SVD 最小二乘刚体变换（Kabsch 算法）** 求解机器人坐标系与基坐标系之间的旋转矩阵 R 和平移向量 t，输出欧拉角（XYZ）、KUKA OAT 角、四元数与逐点配准误差。

## 特性

- 🌐 **纯静态网页演示**：`index.html` 单文件，无需后端，可直接部署 GitHub Pages
- 🐍 **浏览器内运行真实 Python**：通过 Pyodide（WebAssembly）加载 NumPy，算法在页面中实际执行，可在页面底部查看完整源码
- 🤖 **三种机器人格式**：KUKA `.dat`（E6POS）/ FANUC `.ls` / ABB `.mod`（MoveL、MoveP）
- 📊 **误差可视化**：双坐标系 3D 点云对比（机器人坐标系 / 基坐标系叠加），误差热力着色 + 残差连线，逐点误差表与 RMSE / MAE 统计
- ✅ **超差判定**：阈值 ±1 mm，自动给出「合格 / 结果超差」
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
```

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
| 三坐标 `.txt` | `P1,164.374,-36.673,215.159`（逗号分隔，允许表头行） |

机器人点数与三坐标点数必须一致且 ≥ 3 点，点与点一一对应。

## License

MIT
