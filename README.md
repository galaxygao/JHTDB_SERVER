# JHTDB 全周期域速度场流水线

面向 JHTDB `isotropic1024coarse` 的本地、串行、可恢复数据流水线。项目只下载原始速度场，在本地构造完整周期域，持久化全部速度梯度，并基于经过验证的数据计算滤波速度、滤波梯度、work 和 regime。

> 当前状态（2026-08-26）：`time_index=1` 的速度场已完成 `512/512` tile 下载并标记为 `auto_validated`。26 项离线测试通过。梯度和物理计算由用户自行启动，Codex 不代替用户运行数据任务。

旧版说明保存在 [`old readme.md`](old%20readme.md)，重构原则和删除项见 [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)。

## 目录

- [1. 流水线概览](#1-流水线概览)
- [2. 快速开始](#2-快速开始)
- [3. 运行中旧梯度进程的升级兼容](#3-运行中旧梯度进程的升级兼容)
- [4. 数据集、坐标与时间](#4-数据集坐标与时间)
- [5. 数据管理与耦合关系](#5-数据管理与耦合关系)
- [6. 托管梯度](#6-托管梯度)
- [7. 物理计算](#7-物理计算)
- [8. 正确性测试与真实数据审计](#8-正确性测试与真实数据审计)
- [9. 只读 GUI](#9-只读-gui)
- [10. 内存与磁盘预算](#10-内存与磁盘预算)
- [11. JHTDB 使用约束](#11-jhtdb-使用约束)
- [12. CLI 参考](#12-cli-参考)
- [13. 项目结构](#13-项目结构)
- [14. 私有 GitHub 仓库](#14-私有-github-仓库)
- [15. 官方资料](#15-官方资料)

## 1. 流水线概览

```text
JHTDB GetCutout（仅 velocity）
        │
        ▼
完整速度快照 velocity.zarr
        │  自动完整性验证 + manifest
        ▼
9 个周期谱梯度 gradients.zarr
        │  全域无散验证
        ▼
滤波速度 + 其谱梯度 filtered.zarr
        │
        ▼
work_full + work_resolved + regime
        │
        ▼
只读 Web GUI
```

项目分为三个板块：

1. **Download**：从 JHTDB 严格串行下载一个明确的时间索引，只请求速度，不请求压力、梯度、Hessian 或 Laplacian。
2. **Assemble, Validate & Visualize**：把 512 个空间 tile 无重叠地写入唯一完整周期域，自动验证后通过只读 GUI 查看。
3. **Gradient & Physics**：计算并管理 9 个全域梯度，通过无散验证后一次性生成滤波速度及其谱梯度，再计算 work 和 regime。

## 2. 快速开始

所有命令均在项目根目录执行。每次只处理一个明确的 `time_index`。

### 2.1 配置本地 token

```powershell
# 明文输入，保存到当前 Windows 用户的 Credential Manager
python -m jhtdb_pipeline auth set

# 默认只显示是否已配置
python -m jhtdb_pipeline auth status

# 显式显示明文 token
python -m jhtdb_pipeline auth status --show-token
```

token 不写入源码、配置、环境变量、manifest、GUI 或 Git。只有 `auth status --show-token` 会主动显示它。

### 2.2 查看计划和资源预算

```powershell
python -m jhtdb_pipeline plan --time-index 1 --config configs/pipeline.yaml
```

该命令只读取配置并估算下载、梯度和物理阶段资源，不下载数据。

### 2.3 完整处理一个时间帧

```powershell
# 1. 小型在线接口检查：只请求 8³ 速度数据，不登记为完整快照
python -m jhtdb_pipeline smoke --time-index 1 --config configs/pipeline.yaml

# 2. 下载并自动验证完整速度快照
python -m jhtdb_pipeline download --time-index 1 --config configs/pipeline.yaml

# 3. 如有需要，单独重跑速度完整性验证
python -m jhtdb_pipeline validate --time-index 1 --config configs/pipeline.yaml

# 4. 托管 9 个梯度、自动验证无散性，并生成滤波速度及其梯度
python -m jhtdb_pipeline gradient --time-index 1 --config configs/pipeline.yaml

# 5. 用真实数据中心 32³ 小块进行 FD8/FFT 梯度审计
python -m jhtdb_pipeline audit-gradient --time-index 1 --size 32 --config configs/pipeline.yaml

# 6. 如需单独复核，可再次流式验证完整周期域无散性
python -m jhtdb_pipeline validate-divergence --time-index 1 --config configs/pipeline.yaml

# 7. 读取托管的原始/滤波场，计算 work 与 regime
python -m jhtdb_pipeline compute --time-index 1 --config configs/pipeline.yaml

# 8. 打开只读 GUI
python -m jhtdb_pipeline gui --config configs/pipeline.yaml
```

`download` 完成 512 个 tile 后会自动调用速度完整性验证。新版 `gradient` 在 9 个原始梯度完成后自动执行全域无散性验证，只有通过后才生成 `filtered.zarr`。`compute` 不访问 JHTDB，也不重新计算原始梯度、滤波速度或滤波梯度。

## 3. 运行中旧梯度进程的升级兼容

如果 `gradient` 在本次源码更新之前已经启动：

- 正在运行的 Python 进程使用启动时加载到内存的旧代码；磁盘上的 `.py` 修改不会热更新或改变其当前 FFT。
- 在旧进程结束前，不要启动新版 `status`、`gradient`、`compute` 或 GUI，避免写入期间触发 SQLite schema 升级。
- 旧进程可以继续完成；它写出的 Zarr 梯度仍记录输入速度 manifest hash。
- 如果旧版本早于 `128³` chunk 回读优化，校验阶段峰值 RAM 可能约为 `1–1.5 GiB`，但 scratch 仍约为 4 GiB，不是旧物理实现的 60 GiB。

旧进程结束后，再运行一次相同命令：

```powershell
python -m jhtdb_pipeline gradient --time-index 1 --config configs/pipeline.yaml
```

新版会执行以下兼容接管：

1. 比较梯度 Zarr 中的输入速度 manifest hash 与当前速度 manifest hash。
2. hash 相同时，只为旧的 verified 梯度回填 catalog 绑定并直接复用，不建立 4 GiB scratch，也不重算。
3. 旧进程中断时，保留已经 verified 的分量，只计算剩余分量。
4. hash 不同时，旧梯度自动失效，9 个分量重置为 `planned`。

如果被中断的是旧版 `compute`：

- raw 速度和 `gradients.zarr` 不受影响；
- 旧 `physics.zarr` 中未完成的 `velocity_bar`/`gradient_bar` 会被新版忽略；
- 新版 `gradient` 把验证后的滤波场写入独立 `filtered.zarr`，不会误接管缺少新算法签名的旧缓存；
- 旧 `scratch/t000001` 会在用户下一次启动新版 `compute` 时由程序按精确路径重新创建。

## 4. 数据集、坐标与时间

目标数据集固定为 JHTDB `isotropic1024coarse`：

| 项目 | 约定 |
|---|---|
| 空间网格 | `1024 × 1024 × 1024` |
| 物理域 | `[0,2π)³` |
| 边界条件 | x、y、z 均周期 |
| 原始变量 | `velocity = (ux,uy,uz)` |
| 下载和存储精度 | `float32` |
| 存储时间间隔 | `0.002` |
| JHTDB cutout 索引 | 1-based 闭区间 |
| 本地数组索引 | 0-based 半开区间 |

时间映射为：

```text
physical_time = (time_index - 1) × 0.002
```

| `time_index` | 物理时间 |
|---:|---:|
| 1 | 0.000 |
| 2 | 0.002 |
| 3 | 0.004 |
| 4 | 0.006 |

空间坐标为：

```text
x_i = i × 2π / 1024,  i = 0,…,1023
```

本地数据仓不保存索引 1024，也不复制 `2π` 周期端点。首尾网格面是周期相邻面，但数值不要求相等。

### 4.1 轴与分量顺序

Giverny cutout 返回：

```text
(z, y, x, component)
```

本项目统一转换为：

```text
(component, z, y, x)
```

分量标签固定为：

```text
component 0 = ux
component 1 = uy
component 2 = uz
```

## 5. 数据管理与耦合关系

数据根目录配置为：

```text
C:\Users\gao\JHTDB_DATA
```

主要内容：

```text
JHTDB_DATA/
├── catalog.sqlite
├── raw/
│   └── velocity.zarr
├── derived/
│   ├── gradients.zarr
│   └── physics.zarr
├── manifests/
├── qa/
└── scratch/
```

### 5.1 速度快照

```text
raw/velocity.zarr/
└── t000001/
    └── velocity  (3,1024,1024,1024) float32
```

axis order 为 `(component,z,y,x)`，chunk 为 `(3,128,128,128)`。每个快照由 512 个互不重叠的 `128³` tile 组成。

tile 只有在写入、回读、shape/dtype/有限值检查和 SHA-256 全部通过后才标记为 `verified`。已验证 tile 在重跑时跳过，因此下载支持断点续传且不会生成第二份完整 raw 数据。

### 5.2 严格耦合键

梯度与速度通过以下身份严格耦合：

```text
(dataset, time_index, velocity_manifest_hash)
```

只有状态为 `auto_validated` 且具有 manifest hash 的速度快照才能进入梯度阶段。速度 manifest 发生变化时，所有相关梯度和物理缓存自动失效。

## 6. 托管梯度

“托管梯度”是本项目的数据管理术语，表示梯度不是临时数组，而是正式保存、校验、登记并可复用的数据产品。

定义为：

```text
gradient[i,j,z,y,x] = ∂u_i / ∂x_j
```

标签对应关系：

| `i` | 速度分量 | `j` | 求导方向 |
|---:|---|---:|---|
| 0 | `ux` | 0 | x |
| 1 | `uy` | 1 | y |
| 2 | `uz` | 2 | z |

因此 9 个分量是：

```text
∂ux/∂x  ∂ux/∂y  ∂ux/∂z
∂uy/∂x  ∂uy/∂y  ∂uy/∂z
∂uz/∂x  ∂uz/∂y  ∂uz/∂z
```

存储结构：

```text
derived/gradients.zarr/
└── t000001/
    └── gradient  (3,3,1024,1024,1024) float32
```

每个 `(i,j)` 分量独立记录：

- dataset、time index 和物理时间；
- 输入速度 manifest hash；
- `planned → computing → verified` 状态；
- 字节数和 SHA-256；
- 写入后的 `128³` chunk 回读校验。

生产梯度使用完整周期轴上的一维 FFT 谱导数，不使用 JHTDB 微分接口，也不把局部块误当成独立周期域。

原始梯度完成后，`gradient` 自动运行全域无散性验证。通过后，对每个速度分量一次性生成：

```text
velocity_bar   = inverse_FFT(H(k) * FFT(velocity))
gradient_bar_j = inverse_FFT(i*k_j * FFT(velocity_bar))
```

滤波梯度从滤波速度求谱导数，不再对 9 个原始梯度分别执行三维滤波。所有 12 个字段逐一写入、回读、计算 SHA-256，并支持按字段断点恢复：

```text
derived/filtered.zarr/
└── t000001/
    ├── velocity_bar  (3,1024,1024,1024) float32
    └── gradient_bar  (3,3,1024,1024,1024) float32
```

`filtered_t000001.json` 同时绑定速度 manifest、原始梯度 manifest、`sigma_grid` 和滤波算法版本。任一输入或算法改变时，旧缓存自动失效。

## 7. 物理计算

### 7.1 定义

项目保留以下定义：

```text
a_i            = u_j ∂_j u_i
bar(a_i)       = GaussianFilter(u_j ∂_j u_i)
a_resolved_i   = bar(u_j) ∂_j bar(u_i)

work_full      = bar(u) · bar(a)
work_resolved  = bar(u) · a_resolved
```

Q1–Q4 由 `work_full` 和 `work_resolved` 的符号组合划分，接近零的点标记为 `uncertain`。

### 7.2 可复用派生量

```text
derived/physics.zarr/
└── t000001/
    ├── work_full       (1024,1024,1024) float32
    ├── work_resolved   (1024,1024,1024) float32
    └── regime          (1024,1024,1024) uint8
```

物理计算读取 `gradients.zarr` 和 `filtered.zarr`，只从托管场构造 acceleration。原始 acceleration 的滤波不能由滤波速度与滤波梯度的乘积替代，因此仍对 3 个 acceleration 分量执行周期谱 Gaussian；随后完成逐点乘法、分量求和和 regime 分类。acceleration 按速度分量流式计算，不长期保存多个 12 GiB 向量中间场。

旧实现对 3 个速度、9 个梯度和 3 个 acceleration 分别做三维滤波，共 45 次全域轴向变换。新实现使用 9 次速度滤波轴变换、9 次滤波速度求导和 9 次 acceleration 滤波，共 27 次，减少 40%。

## 8. 正确性测试与真实数据审计

### 8.1 离线测试

当前 26 项测试全部通过：

```powershell
python -m unittest discover -s tests -v
```

这些测试只使用解析场、临时 SQLite 和小型临时 Zarr，不访问 JHTDB，也不读写正式数据目录。

主要覆盖：

- 唯一编码 cutout 的 `(z,y,x,component)` → `(component,z,y,x)` 转换；
- JHTDB 1-based 区间与本地 0-based 切片；
- `[0,2π)` 坐标和无重复周期端点；
- 三个方向采用不同波数、三个速度分量采用不同系数的解析周期场；
- 全部 9 个 `∂u_i/∂x_j` 的轴和分量对应；
- 小块 FD8 与 FFT 梯度 relative RMS；
- `32³` 合成 Zarr 的速度 → 梯度 → FD8 audit → QA JSON 端到端流程；
- 完整小型合成域的无散场通过测试和可压缩场拒绝测试；
- 小型 Zarr 的速度 → 原始梯度 → 无散验证 → 滤波速度/梯度 → work 端到端流程；
- catalog manifest 变化后的梯度失效与旧 verified 梯度兼容接管；
- Zarr chunk 写入、回读和 SHA-256；
- GUI 只读取二维平面，不加载完整 4 GiB 梯度分量。

### 8.2 真实数据 FD8 审计

完整 9 梯度变为 `verified` 后运行：

```powershell
python -m jhtdb_pipeline audit-gradient --time-index 1 --size 32 --config configs/pipeline.yaml
```

默认读取中心 `32³` core 和 4 点 halo，用八阶中心有限差分独立比较已保存的 FFT 梯度。输出包括：

- 9 个梯度各自的 FFT RMS、FD8 RMS；
- absolute 和 relative difference RMS；
- cosine similarity；
- FFT/FD8 divergence RMS；
- 总体汇总和轴映射元数据。

报告保存为：

```text
qa/gradient_audit_t000001.json
```

FD8 与谱导数是不同数值算子，真实湍流场的差异不应被要求为零。判断轴和分量是否正确时，重点看 9 项 cosine similarity 是否接近 1，并结合 relative RMS 和 divergence RMS。

### 8.3 全域无散性验证

小块审计通过后，对完整 `1024³` 周期域运行：

```powershell
python -m jhtdb_pipeline validate-divergence --time-index 1 --config configs/pipeline.yaml
```

该命令不重新求导，也不执行 FFT。它流式读取托管梯度的三个对角分量，并在全部网格点计算：

```text
divergence = dux/dx + duy/dy + duz/dz
```

报告包含 divergence mean、mean absolute、RMS、最大绝对值及其零基坐标。用于判定的两个无量纲指标为：

```text
relative_divergence_rms
    = RMS(divergence)
      / RMS(sqrt((dux/dx)^2 + (duy/dy)^2 + (duz/dz)^2))

relative_maximum_divergence
    = max(abs(divergence))
      / max(sqrt((dux/dx)^2 + (duy/dy)^2 + (duz/dz)^2))
```

默认通过阈值在配置中明确给出：

```yaml
validation:
  divergence_relative_rms_max: 1.0e-4
  divergence_relative_max_max: 1.0e-3
```

RMS 门槛验证全域总体误差，maximum 门槛防止少量异常点被全域平均掩盖。报告原子写入 `qa/divergence_t000001.json`，并同时绑定 velocity manifest hash 和 gradient manifest hash。任一指标超出阈值时命令返回非零退出码；`gradient` 不会生成滤波场，`compute` 也会拒绝继续。

## 9. 只读 GUI

```powershell
python -m jhtdb_pipeline gui --config configs/pipeline.yaml
```

当前页面包括：

- **原始速度**：选择时间、`ux/uy/uz`、法向和切片索引；
- **速度梯度**：选择 `u_i` 和求导方向 `x_j`，查看对应托管梯度；
- **滤波速度与梯度**：查看 `filtered.zarr` 中的 `velocity_bar` 和 `gradient_bar`；
- **Work 与 regime**：查看 `work_full`、`work_resolved`、Q0–Q4 regime、阈值和全域占比；
- **质量报告**：查看速度 QA、FD8 梯度审计、全域无散性和物理计算报告。

GUI 使用 Zarr `mode="r"`，不修改 raw、gradient 或 physics 数据，不保存 accept/reject，也不构成人工门禁。未标记为 `complete` 的物理结果不会显示。

## 10. 内存与磁盘预算

以下均为单个 `1024³` 时间帧的保守预算：

| 阶段 | 长期数据上限（未压缩） | scratch/memmap | 保守进程峰值 RAM | 含 40 GiB 余量的最低空闲空间 |
|---|---:|---:|---:|---:|
| 原始速度下载 | 12 GiB | 很小 | 取决于单 tile，通常低于 1 GiB | 52 GiB |
| 9 个托管梯度 | 36 GiB | 4 GiB | 约 256 MiB | 80 GiB |
| 滤波速度 + 滤波梯度 | 48 GiB | 12 GiB | 约 384 MiB | 100 GiB |
| work + regime | 9 GiB | 16 GiB | 约 384 MiB | 65 GiB |
| `32³` FD8 audit | 仅 JSON | 无 | 数组约 3–5 MiB | 可忽略 |
| 全域散度验证 | 仅 JSON | 无 | 单个 `128³` chunk 约 56 MiB | 可忽略 |

说明：

- scratch 是磁盘 memmap 大小，不等于常驻 RAM。
- 梯度、滤波预处理和物理 FFT slab 默认宽度为 4，单个输入块约 16 MiB。
- Zarr 写入和回读按 `128³` chunk 执行；单个 float32 标量 chunk 为 8 MiB。
- 每个全域分量结束后显式 flush，限制脏页累积。
- Windows 文件缓存可能额外显示约 `0.5–2 GiB` 系统缓存，实际值由系统内存压力决定。
- 压缩率取决于数据，容量预检不假设一定能压缩。

## 11. JHTDB 使用约束

本项目遵守用户提供的 JHTDB 使用建议：

- 正式规则网格下载只使用 `GetCutout`。
- 本地 `GetCutout` 单次上限为 3 GB；本项目使用 `128³` tile，每个速度响应未压缩约 24 MiB。
- 一个完整速度快照包含 `8 × 8 × 8 = 512` 个 tile。
- 任意时刻只允许一个 JHTDB 请求，不提供并发开关。
- `GetData` 不用于生产下载，只保留给未来的小规模独立 QA。
- 不提供自动遍历整个时间范围的 crawler；每个命令只接受一个明确时间索引。
- JHTDB 明确不鼓励抓取大量完整 3D 场。多个全域时间帧应谨慎规划，并考虑 SciServer 或自行运行模拟。

## 12. CLI 参考

| 命令 | 作用 | 是否访问 JHTDB |
|---|---|---:|
| `auth set/status/delete` | 管理 Windows Credential Manager 中的 token | 否 |
| `plan --time-index N` | 显示下载和资源计划 | 否 |
| `smoke --time-index N` | 请求一个 `8³` 速度小块 | 是 |
| `download --time-index N` | 下载并自动验证一个完整速度快照 | 是 |
| `validate --time-index N` | 重跑本地速度完整性验证 | 否 |
| `gradient --time-index N` | 托管 9 个梯度、验证无散性并生成滤波速度与梯度 | 否 |
| `audit-gradient --time-index N` | 小块 FD8/FFT 梯度审计 | 否 |
| `validate-divergence --time-index N` | 流式验证完整周期域无散性 | 否 |
| `compute --time-index N` | 使用托管的原始/滤波场计算 work 与 regime | 否 |
| `status` | 查看速度 tile 和梯度 catalog 状态 | 否 |
| `gui` | 启动只读 Streamlit GUI | 否 |

查看当前状态：

```powershell
python -m jhtdb_pipeline status --config configs/pipeline.yaml
```

## 13. 项目结构

```text
README.md
old readme.md
MIGRATION_GUIDE.md
pyproject.toml
configs/
└── pipeline.yaml
src/jhtdb_pipeline/
├── __main__.py
├── auth.py
├── catalog.py
├── cli.py
├── config.py
├── dashboard.py
├── gradients.py
├── jhtdb.py
├── physics.py
├── planning.py
├── store.py
└── validation.py
tests/
├── test_auth.py
├── test_catalog_store.py
├── test_coordinates.py
├── test_dashboard.py
├── test_gradients.py
├── test_physics.py
└── test_planning.py
```

大型 `raw/`、`derived/`、`scratch/` 和 GUI cache 均位于仓库外，不提交到 Git。

## 14. 私有 GitHub 仓库

私有仓库可以使用 HTTPS 地址克隆：

```powershell
git clone https://github.com/galaxygao/JHU_DATA.git
```

URL 本身不授予访问权限。克隆者仍需具有仓库权限，并通过 Git Credential Manager 或 PAT 完成 GitHub 身份验证。

## 15. 官方资料

- [JHTDB Forced Isotropic Turbulence 数据集说明](https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic)
- [JHTDB isotropic1024 数据说明 PDF](https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic.pdf)
- [JHTDB Python Local：GetCutout 获取原始网格数据](https://turbulence.idies.jhu.edu/database/local/python)
- [JHTDB token 与数据库访问](https://turbulence.idies.jhu.edu/database)
- [官方 Giverny 仓库](https://github.com/sciserver/giverny)
