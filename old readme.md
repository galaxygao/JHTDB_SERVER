# JHTDB acceleration regimes（旧版）

本项目使用 JHTDB `isotropic1024coarse` 数据库，在真实 DNS 网格点上获取速度、`fd8noint` 和 `fd6noint` 速度梯度，然后计算局部 Gaussian 滤波后的

\[
\bar a_i=\overline{u_j\partial_j u_i},
\qquad
\overline{\bar a}_i=\bar u_j\partial_j\bar u_i,
\]

以及

\[
W_{\rm full}=\bar{\boldsymbol u}\cdot\bar{\boldsymbol a},
\qquad
W_{\rm res}=\bar{\boldsymbol u}\cdot\overline{\bar{\boldsymbol a}}.
\]

程序根据两个 work 的符号划分 Q1–Q4，并用 FD6/FD8 分类是否一致来生成更保守的 `regime_robust`。

> 重要限制：这里使用数据库局部有限差分梯度和局部 Gaussian 滤波，不是完整周期域 Fourier 求导或全域 spectral filter。数据库说明指出局部梯度相对 DNS 全局谱梯度约有 7% RMS 误差，因此程序保留 `boundary/uncertain` 类别并报告 divergence。

## 1. 最短运行流程

在 PowerShell 中进入项目目录：

```powershell
cd C:\Users\gao\Desktop\Sync_File\26FA\JHU_DATA
```

首次在新 Python 环境中使用时安装依赖及本项目：

```powershell
python -m pip install -e .
```

依次检查计划、测试接口、下载和计算：

```powershell
python task0.py plan
python task0.py smoke
python task0.py fetch
python task0.py validate
python task0.py compute
python task0.py verify
```

确认 smoke 成功后，也可以用一条命令完成正式下载、校验和计算：

```powershell
python task0.py run
```

当前默认正式任务为 4096 个空间点、100 个时间快照。速度、FD8 梯度和 FD6 梯度合计约 30 个严格串行 API 请求；失败的长时间请求会自动拆成更短时间段。

## 2. 顶层脚本和文件

### `task0.py`

最直接的 Python 主脚本。它调用 `jhtdb_regimes.cli`，支持：

```powershell
python task0.py plan
python task0.py smoke
python task0.py fetch
python task0.py validate
python task0.py verify
python task0.py compute
python task0.py classify
python task0.py report
python task0.py run
```

建议优先使用这个入口，因为命令清晰并且不依赖 PowerShell execution policy。

### `task0.bat`

Windows 批处理入口，内部执行 `python task0.py`。例如：

```powershell
task0.bat plan
task0.bat run
```

它与 `task0.py` 功能相同，只是方便从 `cmd.exe` 或 Windows 快捷方式启动。

### `run_task0.ps1`

PowerShell 入口。它会先切换到项目目录，再运行 Python CLI，并在 Python 返回非零退出码时抛出错误。例如：

```powershell
.\run_task0.ps1 -Command smoke
.\run_task0.ps1 -Command run
```

如果 PowerShell 阻止本地脚本，可直接使用 `python task0.py ...`，不需要修改系统策略。

### `dashboard.py` 与 `dashboard.bat`

逐帧 Web 验证面板入口。`dashboard.py` 是 Streamlit 应用入口，`dashboard.bat` 会在本机 `127.0.0.1:8501` 启动服务。面板只读取已有 raw、derived 和 verification 文件，不访问 JHTDB。

### `configs/task0.yaml`

Task 0 的唯一运行配置。空间块、时间范围、testing-token 批次、梯度方法、滤波尺度、regime 容差和输出位置都在这里修改。详细字段见第 6 节。

### `pyproject.toml`

Python 项目和依赖定义，负责安装：

- `givernylocal`：JHTDB 官方本地 Python 客户端；
- `numpy`、`scipy`：数组与数值计算；
- `pandas`：解析 JHTDB DataFrame；
- `PyYAML`：读取配置。

安装后还会提供等价命令：

```powershell
jhtdb-regimes plan configs/task0.yaml
```

### `task 0.md`

当前方案的完整科学与工程设计，包括为什么不使用局部块 FFT、滤波定义、halo/core、regime 定义、误差边界、测试要求和验收标准。运行前若需要理解计算假设，应先阅读这个文件。

### `test.md`

根据项目书第 3.1、3.2 节和 JHTDB 资料形成的前期技术总结，记录原始完整域 Fourier 路线、物理量定义、符号问题及数据库梯度与谱梯度的区别。它是设计依据，不被程序直接读取。

### `acceleration_regimes (1).md`

早期 acceleration/regime 说明或参考笔记。它不参与当前程序运行；正式实现以 `task 0.md` 和 `configs/task0.yaml` 为准。

### `.gitignore`

排除 Python 缓存、安装元数据以及可重新生成的大型下载/计算数据，避免误提交。

## 3. `src/jhtdb_regimes/` 源码

这里是实际执行下载和物理计算的核心代码。

### `__init__.py`

定义 Python package 和版本号。它本身不执行下载或计算。

### `cli.py`

命令行调度器，解释 `plan`、`smoke`、`fetch`、`validate`、`compute`、`classify`、`report` 和 `run`，然后调用对应模块。

各命令行为：

| 命令 | 是否访问网络 | 作用 |
|---|---:|---|
| `plan` | 否 | 显示点数、时间数、批次数、预计请求数和原始数组大小 |
| `smoke` | 是 | 查询 8 点 × 2 时间的速度和 FD8 梯度，检查 token、时间和列名 |
| `fetch` | 是 | 下载 100 snapshots 的 velocity、FD8 和 FD6，写入 cache 和 raw 文件 |
| `validate` | 否 | 检查 raw 的时间、shape、有限值和必要数组 |
| `compute` | 否 | 从 raw 计算滤波量、work、divergence、regimes 和报告 |
| `verify` | 否 | 独立复算 FD6/FD8、直接三维滤波、派生代数和 regime，生成验证报告 |
| `classify` | 否 | 当前与 `compute` 等价，重新计算并分类 |
| `report` | 否 | 从已有 derived 文件重新生成 JSON/Markdown 报告 |
| `run` | 是 | 按 `plan → fetch → validate → compute` 执行完整流水线 |

### `config.py`

读取和验证 `configs/task0.yaml`，生成不可变的 `TaskConfig`。主要安全检查包括：

- testing-token 单批不得超过 4000 点；
- block、halo 和 core 尺寸必须有效；
- 当前实现要求每个方向的 halo 等于 filter support radius；
- block 必须位于全局 1024³ 数组表示范围内；
- 时间步长和时间分块必须为正。

### `grid.py`

负责空间和时间索引：

- 把 `(i,j,k)` 转为精确的 JHTDB 坐标；
- 生成 `(x,y,z)` API point 数组；
- 把 4096 点拆成 4000 + 96 两个请求；
- 把 100 个时间拆成 5 组；
- 将 API 行数据恢复为规则 `(z,y,x)` 网格；
- 保证 velocity component 和 gradient derivative axis 不发生置换。

数组使用 C-order，空间展开时 `x` 最快变化，然后是 `y`、`z`。

### `jhtdb_client.py`

JHTDB 下载客户端，是所有网络访问集中发生的位置。主要功能：

- 从 `JHTDB_TOKEN` 环境变量读取个人 token；
- 未设置环境变量时，从官方 metadata 取得内置 testing token；
- token 只存在内存中，不写入项目文件或日志；
- 所有查询使用官方 `givernylocal.getData`；
- 网络请求严格串行，没有线程池、多进程或异步并发；
- 按列名解析 `ux/uy/uz` 和九个 `du?d?` 梯度分量；
- 校验请求时间与返回时间是否相同；
- 检查点数、列数、NaN 和 Inf；
- 每个成功请求立即保存断点；
- 请求失败时重试，仍失败则自动拆分时间段；
- 合并 point/time 批次并生成 `task0_raw.npz` 与 manifest。

官方 smoke test 已确认当前返回列为：

```text
velocity: ux, uy, uz
gradient: duxdx, duxdy, duxdz,
          duydx, duydy, duydz,
          duzdx, duzdy, duzdz
```

### `physics.py`

只做本地数值计算，不访问网络。包含：

- 生成归一化一维 Gaussian kernel；
- 对最后三个空间轴执行 separable valid convolution；
- 只输出 halo 内侧的 core，不使用 `wrap`、`reflect` 或最近值填充；
- 计算 `a_i = u_j G_ij`；
- 计算 `velocity_bar`、`gradient_bar`、`a_bar` 和 `a_barbar`；
- 计算 `work_full` 和 `work_resolved`；
- 计算梯度张量 trace，即 divergence；
- 按两个 work 的符号产生 Q1–Q4；
- 将 near-zero、NaN/Inf 和 FD6/FD8 不一致点标成 0，即 uncertain。

regime 编码：

| 编码 | 类别 | `work_full` | `work_resolved` |
|---:|---|---:|---:|
| 0 | boundary/uncertain | 接近零或不稳定 | 接近零或不稳定 |
| 1 | Q1 | 正 | 正 |
| 2 | Q2 | 正 | 负 |
| 3 | Q3 | 负 | 正 |
| 4 | Q4 | 负 | 负 |

### `pipeline.py`

连接 raw data 和物理计算，负责：

- `validate_raw()`：验证下载结果；
- `compute()`：运行 FD8 和 FD6 两套物理计算；
- 生成 `regime_primary`、`regime_audit` 和 `regime_robust`；
- 保存 `task0_derived.npz`；
- 计算 regime occupancy、work、divergence 和 FD6/FD8 差异；
- 生成 JSON 和 Markdown 报告。

### `verify.py`

对已经完成的 raw/derived 数据执行独立审计，不访问 JHTDB：

- 验证 raw SHA-256、100 个时间、4096 个网格 index 和所有有限值；
- 使用明确的六阶/八阶中心差分系数，从下载速度独立重建九个梯度分量；
- 将本地重建梯度与数据库 `fd6noint`、`fd8noint` 逐点比较；
- 使用独立的完整三维 kernel direct convolution 检查生产代码的 separable Gaussian convolution；
- 独立复算 `a_barbar`、两个 work 和三套 regime；
- 生成 `task0_verification.json` 与 `task0_verification.md`。

运行：

```powershell
python task0.py verify
```

当前完整 cycle 的验证结果为 PASS。主要结果：

| 检查 | relative RMS |
|---|---:|
| database FD8 vs velocity local FD8 | `1.55e-7` |
| database FD6 vs velocity local FD6 | `1.50e-7` |
| database FD6 vs database FD8 | `1.36e-2` |
| direct 3-D filter vs separable filter | `2.86e-8` |

前两项接近 float32 舍入误差，说明网格顺序、分量顺序、间距和有限差分 stencil 一致。第三项是不同阶数有限差分之间的实际差异，不是程序错误。

## 4. `tests/` 测试程序

运行全部离线测试：

```powershell
python -m unittest discover -s tests -v
```

### `test_grid.py`

验证：

- index 到物理坐标的转换；
- 第一个点和相邻 x 点的顺序；
- 4096 点必须拆成 4000 + 96；
- point rows 恢复为 velocity/gradient 网格时不交换轴。

### `test_columns.py`

用故意打乱顺序的列验证 velocity 和 gradient 解析。确保程序根据名字而不是默认位置映射 `G_ij = ∂_j u_i`；缺少列时必须失败。

### `test_physics.py`

验证：

- Gaussian kernel 权重和为 1；
- 常数场滤波后不变；
- 16³ block 经 valid filter 后严格得到 8³ core；
- acceleration 缩并使用正确的 derivative axis；
- divergence 是梯度张量的 trace；
- Q1–Q4、boundary 和 robust regime 规则；
- 完整派生计算中各数组 shape 正确。

### `test_online.py`

真正访问 JHTDB 的在线 smoke test。为了避免每次运行测试都消耗 testing-token 请求，它默认跳过。显式运行：

```powershell
$env:JHTDB_ONLINE = '1'
python -m unittest discover -s tests -p 'test_online.py' -v
Remove-Item Env:JHTDB_ONLINE
```

通常直接执行 `python task0.py smoke` 更方便。

## 5. `data/` 运行数据

这些目录由程序自动创建。

### `data/cache/`

保存每一个成功 API 请求的压缩 `.npz` 断点，以及 `givernylocal` 自身的输出目录。

- 应在下载尚未完成时保留；
- 重新运行 `fetch` 会验证并复用有效断点；
- 删除后不会损坏代码，但下次必须重新请求对应数据；
- 缓存文件名包含数据集、空间块和时间列表的配置指纹；修改这些配置后不会误用旧任务断点。

### `data/raw/`

正式下载完成后包含：

```text
task0_raw.npz
task0_raw.manifest.json
```

`task0_raw.npz` 的重要数组：

| 数组 | shape | 含义 |
|---|---|---|
| `times` | `(time,)` | API 返回并校验后的时间 |
| `indices_ijk` | `(point,3)` | 全局网格 index |
| `velocity` | `(time,3,z,y,x)` | 速度 |
| `gradient_primary` | `(time,3,3,z,y,x)` | FD8 梯度 |
| `gradient_audit` | `(time,3,3,z,y,x)` | FD6 梯度 |

manifest 保存数据集、空间块、梯度方法、串行策略和 raw 文件 SHA-256，但不保存 token。

raw 文件需要 API 才能重建，因此完成下载后建议备份。

### `data/derived/`

包含 `task0_derived.npz`，由 raw 数据在本地计算得到。重要数组包括：

```text
velocity_bar
gradient_bar_primary
a_bar
a_barbar
work_full
work_resolved
regime_primary
regime_audit
regime_robust
divergence_primary
divergence_bar_primary
```

derived 文件可以通过 `python task0.py compute` 从 raw 重新生成，不需要再次访问 JHTDB。

### `data/reports/`

- `smoke.json`：最近一次 testing-token smoke test 的数据集、点数、返回时间和列名，不含 token；
- `task0_report.json`：机器可读的完整统计；
- `task0_report.md`：适合直接阅读的 regime 汇总和误差提示。
- `task0_verification.json`：独立验证的全部数值和逐样本滤波误差；
- `task0_verification.md`：独立验证的简明 PASS/FAIL 报告。

报告可由 `python task0.py report` 从 derived 文件重新生成。

## 6. `configs/task0.yaml` 配置说明

### 数据集和全局网格

```yaml
dataset: isotropic1024coarse
variable: velocity
grid_shape: [1024, 1024, 1024]
domain_length: 6.283185307179586
```

这里的 domain length 是 \(2\pi\)。当前程序针对规则周期立方网格设计。

### 空间块

```yaml
block_start_ijk: [504, 504, 504]
block_shape: [16, 16, 16]
halo: [4, 4, 4]
```

16³ block 共 4096 点；每个方向去掉 4 点 halo 后输出 8³ core。局部 block 自身不被当成周期域。

### 时间

```yaml
time:
  start: 0.0
  end: 9.9
  step: 0.1
  chunk_size: 20
```

生成 100 个目标时间，每次优先查询连续 20 个时间。`step` 应选择 coarse 数据存储间隔 0.002 的整数倍，避免不必要的时间插值。

### API

```yaml
api:
  token_env: JHTDB_TOKEN
  use_builtin_testing_token: true
  max_points_per_query: 4000
  retries: 3
  retry_backoff_seconds: 1.0
```

testing token 下不要把 `max_points_per_query` 改到 4000 以上，也不要在外部同时启动多个 `fetch` 进程。

如果以后获得个人 token，可以在当前 PowerShell 会话中设置：

```powershell
$env:JHTDB_TOKEN = '你的个人 token'
```

不要把 token 直接写进 YAML 或提交到版本库。

### 梯度、滤波和 regime

```yaml
gradient:
  primary: fd8noint
  audit: fd6noint

filter:
  kind: local_discrete_gaussian
  sigma_grid: 1.0
  support_radius: 4

regime:
  epsilon_abs: 0.0
  epsilon_rel: 0.001
```

- FD8 是主计算；
- FD6 用于检查符号稳定性；
- `sigma_grid` 单位为网格间距；
- `support_radius=4` 对应 9 点一维 kernel；
- `epsilon_rel` 按 work RMS 设置 near-zero boundary。

### 输出路径

```yaml
paths:
  cache: data/cache
  raw: data/raw/task0_raw.npz
  derived: data/derived/task0_derived.npz
  reports: data/reports
```

相对路径均以项目根目录为基准。

## 7. 完整数据流

```text
configs/task0.yaml
        │
        ├── plan：只计算请求计划
        │
        ├── smoke：8 点 × 2 时间 API 检查
        │
        └── fetch
             │
             ├── data/cache/*.npz
             └── data/raw/task0_raw.npz
                         │
                         ├── validate
                         └── compute
                              │
                              ├── data/derived/task0_derived.npz
                              └── data/reports/task0_report.*
```

## 8. 临时和自动生成文件

以下内容不是手写源码，可以安全删除，但含义不同：

- `__pycache__/`、`*.pyc`：Python bytecode cache，可随时删除；
- `src/jhtdb_regimes.egg-info/`：`pip install -e .` 生成的安装 metadata，删除后可重新安装；
- `data/derived/`：可由 raw 本地重算；
- `data/reports/`：可由 derived 重建；
- `data/cache/`：可删除，但会失去断点续传并可能重新消耗 API 请求；
- `data/raw/`：可删除，但必须重新访问 JHTDB 才能恢复。

## 9. 常见操作

只查看任务规模，不访问网络：

```powershell
python task0.py plan
```

测试 testing token：

```powershell
python task0.py smoke
```

中断后继续下载：

```powershell
python task0.py fetch
```

已有 raw，只重新调整计算或 regime 参数：

```powershell
python task0.py compute
```

只重新生成报告：

```powershell
python task0.py report
```

对下载、导数、滤波和 regime 做完整离线验证：

```powershell
python task0.py verify
```

验证报告位于：

```text
data/reports/task0_verification.md
data/reports/task0_verification.json
```

当前 16³ block 可以验证数据库有限差分 stencil 和 Gaussian 数值实现，但不能直接验证完整的 `derivative(filter(u)) = filter(derivative(u))`。原因是 Gaussian 和 FD8 都各需要 4 点 halo；若最终仍要保留 8³ 验证 core，需要至少下载 24³ block。这个限制不影响已经完成的两项独立检查，但应在论文或报告中明确说明。

检查全部离线测试：

```powershell
python -m unittest discover -s tests -v
```

## 10. 逐帧 Web 可视化验证

启动面板：

```powershell
dashboard.bat
```

也可以直接运行 Python 入口；它会自动转入 Streamlit CLI：

```powershell
python dashboard.py
```

或显式运行：

```powershell
python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

浏览器打开：

```text
http://127.0.0.1:8501
```

面板顶部提供“上一帧”“下一帧”和 frame slider；侧栏可选择 `x/y/z` 切片方向和 8³ core 内的切片位置。页面包括：

1. **总览与完整性**：checksum、时间、index、有限值、所有离线 PASS/FAIL，以及跨帧误差曲线；
2. **速度场**：raw core、filtered velocity 和二者变化；
3. **导数对比**：database FD8/local FD8、database FD6/local FD6 及所有差值；
4. **滤波对比**：生产 separable filter、独立 direct 3-D convolution 和差值，可选择 velocity、gradient 或 acceleration；
5. **Acceleration**：FD8/FD6 的 raw acceleration、`a_bar`、`a_barbar` 及差值；
6. **Work 与 regimes**：两种 work 的 FD8/FD6 对比、primary/audit/robust regime、disagreement 和 uncertain masks；
7. **Divergence**：FD8、FD6、filtered divergence 及差值；
8. **单点随时间**：选择一个 core \((i,j,k)\)，查看该位置 100 帧的速度、梯度、acceleration、work、divergence 和 regime；
9. **跨帧时间序列**：默认只显示同位置误差汇总、divergence、regime disagreement/occupancy 和 uncertain fraction。

Velocity/Gradient/Work RMS 被放入单独的“空间量级汇总”折叠区，不作为正确性判据。它们会把空间变化压成一个数，其中 Work RMS 还会丢失正负号；真正的场验证应查看同一位置的差值图和单点时间序列。

普通场并排图使用共同色标，差值图使用以零为中心的独立对称色标，便于发现局部错误。鼠标悬停可读取切片 index 和数值。
