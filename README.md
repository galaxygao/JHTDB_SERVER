# JHTDB SciServer 全周期域流水线

本项目只在 Johns Hopkins SciServer 上运行。它从 JHTDB 获取 `isotropic1024coarse` 的单帧完整速度场，在完整周期域上完成谱导数与谱高斯滤波，把中心 `512^3` 的速度和梯度，以及全域 `1024^3` 的四个能量场与 regime 保存到 SciServer persistent，并在 Interactive container 中提供只读 GUI。科学数据不下载到本地电脑，也不依赖本地文件系统。

SciServer 的系统结构、容器、Compute Job 和存储卷说明见 [`SCISERVER_SYSTEM_GUIDE.md`](SCISERVER_SYSTEM_GUIDE.md)。本 README 是项目功能、配置、安装、运行、代码结构和验收标准的主入口。

## 1. 项目目的、功能与实现概括

### 1.1 目的

项目解决以下问题：

- 从 JHTDB 可靠地获取一帧完整 `1024^3 × 3` 周期速度场；
- 避免把可重建的完整原始速度和 FFT 工作场长期保存在 persistent；
- 保证 FFT、谱导数和滤波在完整周期域上进行，避免先裁剪造成边界伪影；
- 永久保存中心 `512^3` 的未滤波/滤波速度和梯度，以及全域四个能量场与 regime；
- 在服务器上直接查看正式结果，不下载结果归档到本地；
- 对下载、坐标、物理计算、输出提交和 token 使用建立可审计的验证链。

当前版本一次执行一帧，并可按配置顺序批量计算多个滤波尺度。各尺度复用同一份已验证速度缓存，分别生成独立正式结果。

### 1.2 主要功能

- 严格串行的 JHTDB 大块 cutout，并支持断点续跑；
- scratch 中的完整速度缓存、checksum、manifest 和缓存有效性检查；
- 完整 `1024^3` 周期域上的谱导数与可分离谱高斯滤波；
- 中心 `[256:768)^3` 裁剪；
- 服务器 persistent 中的 Zarr 正式结果；
- 全域无散度检查、shape/dtype/有限值检查和逐字段 SHA-256；
- 可重复运行的全域 S̄ QA、两项判据和四场净总量柱状图；
- 全域 Q1–Q4 的 Cq 分解、stored/LES 双符号报告及四项 closure 自检；
- 全域 Π 正负分拆、净通量/RMS 弱非对称指标及 closure 自检；
- staging 到正式结果的原子提交和 `COMPLETE` 标记；
- Streamlit + Plotly 只读服务器 GUI，按需读取二维切片。

### 1.3 实现方法概括

```text
JHTDB isotropic1024coarse
        │ 8 个 512^3 request，严格串行
        ▼
scratch/velocity_cache.zarr
        │ 每个 request 拆成 64 个 128^3 checksum tile
        │ 完整性、坐标、回读 SHA-256、周期接缝检查
        ▼
完整 1024^3 周期域
        │ FFT 谱导数 + 谱高斯滤波 + work
        ▼
persistent/results/.staging/<result_id>
        │ 速度/梯度写中心 [256:768)^3
        │ W_full/W_res/Π/S̄/regime 写完整 1024^3
        │ schema、有限值、散度、字段 SHA-256 验证
        ▼
persistent/results/<result_id>/ + COMPLETE
        │
        └── Interactive container 中的只读 GUI
```

scratch 只保存可重建的完整速度缓存、FFT memmap 和临时工作区；persistent 保存代码、状态、验证记录、中心速度/梯度，以及全域 regime 和四个能量场。正常成功后，尺度对应的 FFT 工作区会按配置清理；完整速度缓存可在确认不再 backfill 该帧后再删除。

### 1.4 科学计算约定

- 数据集：`isotropic1024coarse`；
- 周期域：`[0, 2π)^3`，不重复周期终点；
- 完整网格：`1024^3`；
- 内部速度轴顺序：`(component, z, y, x)`；
- 梯度轴顺序：`(velocity_component, derivative_component, z, y, x)`；
- 裁剪范围：三个方向均为半开区间 `[256, 768)`；
- `time_index` 从 1 开始，`physical_time = (time_index - 1) × 0.002`；
- `sigma_grid` 是以网格间距为单位的高斯标准差。

谱高斯传递函数为：

```text
G(θ) = exp[-0.5 × (sigma_grid × θ)^2]
```

例如 `sigma_grid=1.0` 表示标准差为 1 个网格间距。当前配置列表 `[1.0,2.0,3.0]` 会依次产生三个独立尺度结果。代码没有使用有限长度的离散高斯核；如果采用“与 top-hat 二阶矩等效”的宽度约定，则 `sigma_grid=1.0` 对应 `Δ_eff = √12 σ ≈ 3.464` 个网格间距。命令行和结果目录记录的仍然是 `sigma_grid`，不是 `Δ_eff`。

当前 work 的代码定义为：

```text
work_full     = Σ_i ū_i × overline[(u_j ∂_j u_i)]
work_resolved = Σ_i ū_i × (ū_j ∂_j ū_i)
```

按照 project description 的式 (1)–(2)，另保存：

```text
tau_ij = overline(u_i u_j) - ū_i ū_j
pi     = Σ_ij tau_ij × ∂_j ū_i
s_bar  = Σ_j ∂_j(Σ_i ū_i tau_ij)
work_full = work_resolved - pi + s_bar
```

这里的 `pi` 严格采用式 (2) 的符号约定。常见 LES 文献定义的正向能量通量 `Pi_conventional = -tau_ij S_ij`，在不可压缩且应力对称时等于这里的 `-pi`。QA 用四场联合 RMS 归一化逐点能量等式残差；绝对 residual RMS 和最大绝对值仍写入报告，只作诊断、不参与 pass。

`regime` 在全周期域根据两种 work 相对于各自全域 RMS 阈值的正负组合编码为 uncertain、Q1、Q2、Q3、Q4。

全域 Cq 使用已保存的 `pi`、`work_full` 和 `work_resolved`，按零阈值符号划分四个完备区域：

```text
Cq_stored = mean(pi * I[(sign(work_full), sign(work_resolved)) in Qq])
Cq_LES    = -Cq_stored
C1 + C2 + C3 + C4 = mean(pi)
```

零值归入非负侧，因此 Q1–Q4 覆盖每个全域点并可做严格四项 closure。用于可视化的 thresholded `regime` 仍可包含 uncertain，但不参与 Cq 划分。`pi=tau:S` 是项目存储符号；常见 LES 正向通量为 `Pi_LES=-tau:S=-pi`。报告同时保存每个区域的点数、体积分数、`pi` 总和、对全域均值的贡献和条件均值。

全域 weak asymmetry 报告把 `pi > 0` 解释为 backscatter、`pi < 0` 解释为 forward cascade，并计算：

```text
Pi_pos_sum + Pi_neg_sum = Pi_sum
asymmetry_index = mean(pi) / rms(pi)
ratio_p99 = abs(mean(pi)) / percentile(abs(pi), 99)
ratio_max = abs(mean(pi)) / max(abs(pi))
```

`asymmetry_index` 直接比较净通量和 Π 涨落强度；其绝对值越小，正负强 patch 抵消越显著。`ratio_p99` 和 `ratio_max` 使用净通量绝对值，分别以 `|Π|` 的 99 百分位和最大值归一化。p99 按线性分位数定义用 float64 插值精确计算，流式保留最高约 1% 候选，不需要第二次读取 `pi`。Cq 与 weak asymmetry 共用一次全域 chunk 扫描，避免重复读取数据。

## 2. 配置、安装与使用

本节集中给出从创建容器到查看结果所需的全部配置和命令。除非特别说明，所有命令都在 SciServer 上执行。

### 2.1 SciServer 平台配置

创建或启动 Interactive Compute container 时使用：

| 项目 | 配置 |
|---|---|
| Compute Image | `SciServer Essentials 4.0` |
| Data Volume | `Turbulence (ceph)` |
| User Volume | `persistent`，读写 |
| User Volume | `scratch`，读写 |
| Python | Essentials 4.0 自带 Python 3.9 |

不要用 Essentials 6.0 执行当前 `isotropic1024coarse` 流程。该数据集仍依赖 Essentials 4.0 中可工作的 legacy `pyJHTDB` runtime；Python 3.12 环境中的 PyPI `pyJHTDB` 占位包会主动报弃用错误。

正式计算使用 shell-command Compute Job；环境准备、检查、smoke test、状态查看和 GUI 使用 Interactive container。Job 必须挂载与 Interactive container 相同的 `Turbulence (ceph)`、persistent 和 scratch。

### 2.2 服务器路径

```text
项目与环境：
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA

状态与日志：
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/state

正式结果：
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/results

可重建中间量：
/home/idies/workspace/Temporary/gaoxingqun/scratch/JHTDB_RUNS

token：
/home/idies/workspace/Storage/gaoxingqun/persistent/.secrets/jhtdb_token
```

项目依据账户 Quotas 页面在 2026-08-28 显示的 100 GB persistent 配额进行规划，并默认保留至少 15 GiB 安全余量。`doctor` 会实测挂载文件系统余量，但提交任务前仍以 SciServer Quotas 页面为账户配额的最终依据。scratch 按平台规则属于约 72 小时生命周期、无备份的临时空间。

### 2.3 流水线配置

主配置文件是 [`configs/pipeline.yaml`](configs/pipeline.yaml)。生产约束由配置加载器再次验证，未知字段会被拒绝。

| 配置组 | 关键配置 | 当前值与作用 |
|---|---|---|
| dataset | `dataset`, `variable` | `isotropic1024coarse`, `velocity` |
| grid | `grid_shape`, `domain_length` | `1024^3`, `2π` |
| time | `stored_time_step` | `0.002` |
| platform | `state_root` | persistent 中的 catalog、manifest、QA、锁和日志 |
| platform | `run_root` | scratch 中的完整速度缓存和 FFT 工作区 |
| platform | `result_root` | persistent 中的正式结果 |
| auth | `token_file` | 项目目录外的 `0600` token 文件 |
| jhtdb | `request_shape` | `512^3`；完整帧 8 个请求，严格串行 |
| jhtdb | `tile_shape` | `128^3`；完整帧 512 个校验 tile |
| jhtdb | `retries`, `backoff_seconds` | 请求重试和指数退避 |
| storage | `compression_level` | Zstd/Blosc 压缩等级 3 |
| storage | `compression_threads` | 8 个压缩线程 |
| storage | safety reserve | persistent 15 GiB、scratch 16 GiB |
| validation | divergence thresholds | 原始和滤波速度各自的相对 RMS `1e-4`、相对最大值 `1e-3` |
| validation | S̄ QA thresholds | 能量等式相对联合 RMS `1e-4`、`S_bar_vs_Pi_net` `1e-2`；不计算 `Σ|S̄|` 绝对归一化判据 |
| validation | `cq_partition_relative_max` | C1–C4 closure 相对 `Σ|pi|` 的残差上限，默认 `1e-12` |
| physics | `sigma_grid` | 高斯标准差列表；默认生产配置为 `[1.0,2.0,3.0]` 个网格间距 |
| physics | `crop_start`, `crop_shape` | `[256,256,256]`, `[512,512,512]` |
| physics | `epsilon_abs`, `epsilon_rel` | regime 不确定区阈值参数 |
| physics | `fft_workers` | 16 个 SciPy FFT worker |
| physics | `fft_slab_width` | 32 层 slab，单输入块约 128 MiB |
| physics | `cleanup_scratch_on_success` | 成功提交后清理该尺度 FFT 工作区 |

`request_shape` 和 `tile_shape` 含义不同：一个 `512^3` request 从 JHTDB 获取约 1.5 GiB 未压缩速度数据，随后被写成 64 个 `128^3` checksum tile。中断后已验证 tile 会保留；若某个大 request 仍有缺块，续跑时只重新请求该大块。它们都不是本地传输分卷。

#### 2.3.1 如何调整计算资源

先在目标容器或 Job 中确认 CPU、内存和磁盘：

```bash
nproc
free -h
df -h /home/idies/workspace/Temporary/gaoxingqun/scratch
```

| 参数 | 调大后的主要效果 | 调整建议 |
|---|---|---|
| `fft_workers` | 使用更多 CPU；也会增加线程调度和内存带宽竞争 | 不要超过 Job 分配的 CPU 核数；依次测试 8、16、32，只有分配到约 32 核时才使用 32 |
| `fft_slab_width` | 减少 FFT 批次和 Python/memmap 循环，通常更快；瞬时 RAM 增加 | `16/32/64` 分别约为 `64/128/256 MiB` 输入块，实际 FFT 临时内存是输入块的数倍；从 32 调到 64 前先检查 RAM |
| `compression_threads` | 加快 Zarr 压缩写入，但占用更多 CPU | 通常设为 CPU 核数的 1/4–1/2；不要和 `fft_workers` 一起盲目设满 |
| `compression_level` | 调大后结果通常更小，但写入更慢 | 速度优先用 1，平衡用当前的 3；不能通过调整它改变物理结果 |
| `sigma_grid` | 增加独立滤波尺度和正式结果数量 | 计算时间、persistent 结果量近似随尺度个数线性增长；顺序执行不增加单尺度 scratch 峰值 |
| `cleanup_scratch_on_success` | `false` 会保留每个尺度约 40 GiB workspace | 批量尺度必须保持 `true`，否则多个尺度的 scratch 会累积 |

推荐起点：

```yaml
# 约 16 CPU 核、内存充足：当前平衡配置
storage:
  compression_level: 3
  compression_threads: 8
physics:
  fft_workers: 16
  fft_slab_width: 32
  cleanup_scratch_on_success: true
```

```yaml
# 写入/速度优先；先确认 slab=64 的瞬时内存可接受
storage:
  compression_level: 1
  compression_threads: 8
physics:
  fft_workers: 16
  fft_slab_width: 64
  cleanup_scratch_on_success: true
```

`persistent_capacity_gb_observed` 只记录 Quotas 页面观测值，两个 `*_safety_reserve_gib` 只控制空间预检；降低它们不会加速。`crop_start/crop_shape`、`request_shape/tile_shape` 是当前生产约束，不能把它们当作性能旋钮。`epsilon_abs/epsilon_rel` 只改变 regime 不确定区阈值，基本不影响运行时间。修改配置后先执行 `plan`，确认单尺度/批量 persistent 容量、40 GiB workspace 和约 52 GiB scratch 峰值。

下载和物理计算使用 Rich 进度行，计数分别显示 request 完成数和谱计算 `step/42`。正式脚本为了保存日志使用 `tee`；Rich 因此可能只在阶段结束时显示最终进度，而不是实时刷新，这不代表进程停止。需要在 Interactive Terminal 实时观察时，可直接运行：

```bash
python -m jhtdb_pipeline single-frame --time-index 1 --config configs/pipeline.yaml
```

直接命令与 `run_stage.sh` 的物理计算相同，但不自动写入带时间戳的日志。任务是否仍在运行可用 `ps -eo pid,etime,stat,%cpu,%mem,cmd | grep '[j]htdb_pipeline'` 检查；同一帧任务仍在运行时不要重复启动。

### 2.4 安装

在 JupyterLab Terminal 中执行：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
bash scripts/bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` 会：

1. 拒绝非 Linux 或不在 SciServer persistent 中的项目路径；
2. 在安装前检查 Essentials 4.0 的 functional legacy JHTDB runtime；
3. 创建带 `--system-site-packages` 的 `.venv`；
4. 以 editable mode 安装项目；
5. 导入检查所有项目依赖并编译 `src`。

脚本不会运行全局 `pip check`。Essentials 4.0 镜像包含与本项目无关的 legacy 包，它们可能报告 TensorFlow/protobuf 等全局 metadata 冲突；项目使用显式 import、测试和 `doctor` 验证实际运行环境。

### 2.5 JHTDB token

推荐把 token 写入项目目录之外的受限文件：

```bash
mkdir -p /home/idies/workspace/Storage/gaoxingqun/persistent/.secrets
read -rsp "JHTDB token: " JHTDB_SECRET
printf '%s' "$JHTDB_SECRET" > /home/idies/workspace/Storage/gaoxingqun/persistent/.secrets/jhtdb_token
unset JHTDB_SECRET
chmod 600 /home/idies/workspace/Storage/gaoxingqun/persistent/.secrets/jhtdb_token
```

token 读取优先级：

1. 当前进程环境变量 `JHTDB_TOKEN`；
2. `pipeline.yaml` 中配置的 token 文件。

检查配置状态时不会显示 token 内容：

```bash
python -m jhtdb_pipeline auth status --config configs/pipeline.yaml
```

token 不得写入 YAML、Git、CLI 参数、日志、QA、manifest 或 GUI。

### 2.6 运行前检查

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
source .venv/bin/activate

python -m jhtdb_pipeline doctor --config configs/pipeline.yaml
python -m unittest discover -s tests -v
python -m jhtdb_pipeline plan --time-index 1 --config configs/pipeline.yaml
python -m jhtdb_pipeline smoke --time-index 1 --config configs/pipeline.yaml
```

- `doctor` 检查操作系统、路径、挂载卷写权限、空间、token、Giverny/pyJHTDB runtime 和 scratch 有效期；
- `unittest` 运行离线单元/集成测试，不访问 JHTDB；
- `plan` 只报告请求数量、空间、裁剪和计算资源计划，不抓取数据；
- `smoke` 真实读取一个 `8^3` 小块，用于确认 token、runtime 和 JHTDB 连接。

四项通过后再启动完整单帧任务。

### 2.7 正式单帧 Compute Job

在 SciServer Jobs 页面选择与 Interactive container 相同的 image 和挂载卷，提交：

```bash
bash -lc 'cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA && source .venv/bin/activate && bash scripts/run_stage.sh single-frame --time-index 1'
```

如果希望在 Interactive Terminal 中实时查看 Rich 进度条，不要经过 `run_stage.sh` 的 `tee`，直接运行 Python 入口：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
source .venv/bin/activate
python -m jhtdb_pipeline single-frame \
  --time-index 1 \
  --config configs/pipeline.yaml
```

省略 `--sigma-grid` 时会按配置顺序处理全部尺度；每个需要计算的尺度显示 `periodic spectral pipeline 0/42` 到 `42/42`。若只想运行一个尺度，可增加例如 `--sigma-grid 1.0`。`run_stage.sh` 的计算行为相同并会保留日志，但由于输出经过 `tee`，Compute Job 页面可能只显示阶段结束时的最终进度行。

需要中断 Interactive Terminal 中的任务时优先按 `Ctrl+C`，并确认旧进程退出后再重启。已校验的 JHTDB 分块和完整 v5 结果会复用；v4 会快速补齐全域 regime；当前尺度未完成的 staging 会重算，但匹配且完整的 temporary 滤波速度仍可自动复用。

完整执行顺序：

```text
doctor → cache → validate-input → process-center → finalize-result
```

运行日志写入：

```text
state/logs/single-frame_<UTC timestamp>.log
```

省略 `--sigma-grid` 时，命令依次计算配置中的全部 `physics.sigma_grid`；提供该参数时只计算指定尺度。相同 `time-index` 和 `sigma_grid` 的完整 v5 结果已经存在时直接复用。完整 v4 会直接读取 persistent 中已有的全域 `work_full/work_resolved`，快速补齐全域 regime，不访问 JHTDB、不做滤波或 FFT。旧 v2/v3 使用 temporary 中已验证的滤波速度或 `velocity_cache.zarr` 补算完整 v5；旧 persistent 结果保留到新版校验通过后才替换。

尺度采用顺序批处理，不会同时启动多个约 52 GiB scratch 峰值的 FFT 流程。v5 每个尺度未压缩约 29 GiB，三个尺度约 87 GiB；100 GB 十进制配额折合约 93.1 GiB，无法再同时满足 15 GiB reserve，因此必须先用 `plan` 和 Quotas 决定尺度数量或扩容。

### 2.8 分阶段运行、排错与状态

正常生产任务优先使用 `single-frame`。需要排错时可在 Interactive container 中分别执行：

```bash
# 构建或续跑完整速度缓存
python -m jhtdb_pipeline cache --time-index 1 --config configs/pipeline.yaml

# 重新验证输入缓存
python -m jhtdb_pipeline validate-input --time-index 1 --config configs/pipeline.yaml

# 完整周期域计算并写入 persistent staging
python -m jhtdb_pipeline process-center --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 校验 staging 并原子提交正式结果
python -m jhtdb_pipeline finalize-result --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 用已有 temporary 数据把 v2/v3 中心结果补算为当前全域 schema
python -m jhtdb_pipeline backfill-full-fields --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 只读已有 v4 全域 work 字段，快速补齐全域 regime
python -m jhtdb_pipeline backfill-full-regime --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 自动选择 v4 快速补齐或 v2/v3 temporary 补算；省略 sigma 时处理配置中全部尺度
python -m jhtdb_pipeline upgrade-result --time-index 1 --config configs/pipeline.yaml

# 对完整当前 schema 的四场重复运行全域 S_bar QA 和柱状图
python -m jhtdb_pipeline qa-sbar --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 只读完整结果的全域 pi/work 字段，计算或重算四项 Cq closure
python -m jhtdb_pipeline compute-cq --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 只读完整结果的全域 pi，单独计算或重算 weak asymmetry
python -m jhtdb_pipeline compute-weak-asymmetry --time-index 1 --sigma-grid 1.0 --config configs/pipeline.yaml

# 查看输入和正式结果状态
python -m jhtdb_pipeline status --config configs/pipeline.yaml
```

CLI 命令汇总：

| 命令 | 是否访问 JHTDB | 功能 |
|---|---:|---|
| `auth status` | 否 | 只报告 token 是否配置及来源 |
| `doctor` | 否 | 平台、挂载、环境、空间和 token 前检 |
| `plan` | 否 | 请求、空间和资源计划 |
| `smoke` | 是，小块 | `8^3` 真实读取 |
| `cache` | 是 | 串行构建或续跑完整速度缓存 |
| `validate-input` | 否 | 完整输入缓存验证 |
| `process-center` | 否 | 全域谱计算并写 staging |
| `finalize-result` | 否 | 字段级验证并提交正式结果 |
| `backfill-full-fields` | 否 | 复用 temporary 数据，把已有 v2/v3 结果补算成当前全域 schema |
| `backfill-full-regime` | 否 | 只读 v4 persistent 全域 work 字段，快速补齐全域 regime |
| `qa-sbar` | 否 | 读取当前全域四场，重算两项 S̄ QA 与净总量柱状图 |
| `compute-cq` | 否 | 逐 chunk 读取全域 pi/work，计算 Q1–Q4 Cq，并在同一次扫描生成 weak asymmetry 报告 |
| `compute-weak-asymmetry` | 否 | 只读全域 pi，计算正负总量/占比、净通量/RMS 指标和 closure |
| `upgrade-result` | 否 | 自动选择 v4 regime 快速补齐或 v2/v3 全场补算 |
| `single-frame` | 按需 | 已有 v4/v5 不访问 JHTDB，并自动补齐新版 S̄/Cq/weak-asymmetry 报告；缺少完整结果时串联单帧流程 |
| `status` | 否 | 输入缓存和正式结果状态 |
| `gui` | 否 | 启动只读服务器 GUI |

### 2.9 服务器 GUI

GUI 在 Interactive container 中运行，不在 Compute Job 中运行。它只扫描 `persistent/results` 中带 `COMPLETE` 的正式结果；正在计算的 `.staging` 目录不会显示。

在新的 Terminal 中执行：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
source .venv/bin/activate
python -m jhtdb_pipeline gui --config configs/pipeline.yaml --port 8501
```

GUI 监听 `0.0.0.0:8501`，支持：

- `velocity` 与 `velocity_bar` 同色标切片对比；
- 9 个 `gradient` 与 `gradient_bar` 分量对比；
- 梯度线性或 SymLog 色标和显示分位数；
- `work_full`、`work_resolved` 和 `regime`；
- 式 (2) 的 `pi`、`s_bar` 同色标切片，以及当前切片的能量等式残差；
- 独立的“全域 S̄ QA”页面：两项判据、阈值、通过状态、四场净总量柱状图和原始报告；
- 独立的“Cq 分解”页面：Q1–Q4 占比、贡献、条件均值、双符号柱状图和四项 closure；
- 独立的“Weak asymmetry”页面：Π 正负总量/占比、净通量、RMS、`ratio_p99`、`ratio_max`、非对称指标和 closure；
- manifest、输入/散度 QA 和 `COMPLETE` 完整性记录；
- `x/y/z` 法向与切片 index 选择。

GUI 每次只从 Zarr 读取选中的二维切片，不把整个约 29 GiB 未压缩结果载入内存。浏览器只接收当前页面需要的可视化数据，不生成或下载结果归档。按 `Ctrl+C` 只停止 GUI，不影响另一个 Terminal 或 Compute Job 中的计算。

需要通过当前 SciServer compute domain 提供的端口入口访问。若该 domain 不提供 Streamlit 端口代理，当前版本不会回退到本地 GUI；应增加服务器端 Jupyter 内嵌 viewer。

## 3. 项目与代码详细解析

### 3.1 目录结构

```text
JHU_DATA/
├── configs/
│   └── pipeline.yaml
├── scripts/
│   ├── bootstrap.sh
│   └── run_stage.sh
├── src/jhtdb_pipeline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── auth.py
│   ├── catalog.py
│   ├── cli.py
│   ├── config.py
│   ├── cq.py
│   ├── dashboard.py
│   ├── doctor.py
│   ├── jhtdb.py
│   ├── physics.py
│   ├── planning.py
│   ├── processing.py
│   ├── sbar_qa.py
│   ├── store.py
│   ├── validation.py
│   └── weak_asymmetry.py
├── tests/
│   ├── test_auth.py
│   ├── test_catalog_store.py
│   ├── test_coordinates.py
│   ├── test_cq.py
│   ├── test_dashboard.py
│   ├── test_doctor.py
│   ├── test_fetch.py
│   ├── test_physics.py
│   ├── test_planning.py
│   ├── test_processing.py
│   ├── test_sbar_qa.py
│   └── test_weak_asymmetry.py
├── dashboard.py
├── pyproject.toml
├── README.md
├── SCISERVER_SYSTEM_GUIDE.md
└── Policies – SciServer.pdf
```

运行时还会创建 `.venv/`、`state/` 和 `results/`，这些不是源代码文件。完整速度缓存和 FFT workspace 只位于 scratch。

### 3.2 配置与运行脚本

| 文件 | 职责 |
|---|---|
| `configs/pipeline.yaml` | 唯一生产配置入口；定义数据集、路径、请求、存储、验证和物理参数 |
| `scripts/bootstrap.sh` | 检查 SciServer 4.0 runtime，创建 `.venv`，安装和导入验证项目 |
| `scripts/run_stage.sh` | Compute Job 入口；当前只接受 `single-frame`，生成带 UTC 时间戳的 persistent 日志 |
| `pyproject.toml` | Python 包 metadata、版本、依赖范围和 `jhtdb-pipeline` console script |

### 3.3 Python 包

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包标识和版本级入口 |
| `__main__.py` | 支持 `python -m jhtdb_pipeline` |
| `auth.py` | 从环境变量或 `0600` 文件安全读取 token；失败时关闭访问 |
| `catalog.py` | SQLite 输入 catalog；记录 snapshot、tile、尝试次数、checksum 和状态 |
| `cli.py` | 定义所有子命令，串联 doctor、fetch、validation、processing、finalization 和 GUI |
| `config.py` | 加载 YAML、拒绝未知字段、验证生产约束、生成运行/结果路径和物理时间 |
| `dashboard.py` | Streamlit + Plotly 只读 GUI；只列出带 `COMPLETE` 的结果并按需读取二维切片 |
| `doctor.py` | 检查 Linux、挂载路径、写权限、空间、scratch 生命周期、依赖版本和 Giverny runtime |
| `jhtdb.py` | SciServer Giverny/legacy cutout 适配、轴转换、smoke test、串行请求、重试和断点续跑 |
| `physics.py` | 谱导数、谱高斯滤波、分轴 slab FFT、memmap、乘积累积和 regime 编码 |
| `planning.py` | 生成 8 个 request、512 个 checksum tile、JHTDB 一基坐标范围和资源计划 |
| `processing.py` | 完整域滤波/梯度/四场计算、中心字段裁剪、复用/backfill、staging 和正式结果提交 |
| `sbar_qa.py` | 流式读取全域四场，计算 S̄ 两项 QA，并写 JSON 与 Plotly 柱状图 |
| `cq.py` | 流式读取全域 pi/work，计算 Q1–Q4 Cq、LES 符号结果和四项 closure |
| `weak_asymmetry.py` | 流式分拆全域 Π 正负总量/占比，计算净通量/RMS 指标、closure 和报告产物 |
| `store.py` | Zarr schema、Blosc/Zstd 压缩、tile 回读 checksum、结果字段哈希和只读打开规则 |
| `validation.py` | 输入覆盖/重叠/有限值/接缝验证、manifest hash 和原子 JSON 写入 |

根目录的 `dashboard.py` 是一个简短的 Streamlit 兼容入口，实际 GUI 实现位于 `src/jhtdb_pipeline/dashboard.py`。正常启动仍使用 CLI 的 `gui` 子命令。

### 3.4 测试文件

| 文件 | 覆盖内容 |
|---|---|
| `test_auth.py` | token 来源、空 token、缺失 token、权限约束和不泄露行为 |
| `test_catalog_store.py` | catalog 唯一性、tile 写入/回读和小型完整 snapshot |
| `test_coordinates.py` | Giverny 轴转换、request/tile 分割、一基 API 范围和周期坐标 |
| `test_dashboard.py` | 中心/全域切片、S̄/Cq 柱状图、颜色范围、SymLog 和 COMPLETE 过滤 |
| `test_doctor.py` | scratch 运行记录过期时拒绝继续 |
| `test_fetch.py` | 大 request 拆分、checksum tile 持久化和断点续跑 |
| `test_physics.py` | 周期谱导数、高斯常量保持、全部梯度轴、流式滤波等价性和 regime |
| `test_planning.py` | 8 个请求、512 个无重叠 tile、严格串行计划和完整覆盖 |
| `test_processing.py` | 小型端到端全域四场、temporary 复用、正式提交和散度失败门槛 |
| `test_sbar_qa.py` | 两项 S̄ 判据、全域净和、零分母 JSON 安全和图表产物 |
| `test_cq.py` | Q1–Q4 完整分区、stored/LES 符号、空区域 JSON 安全和四项 closure |
| `test_weak_asymmetry.py` | Π 正负/零分拆、净通量/RMS 非对称指标、closure 和 JSON/HTML 产物 |

### 3.5 文档与平台资料

| 文件 | 职责 |
|---|---|
| `README.md` | 项目目的、统一操作入口、代码结构和验收标准 |
| `SCISERVER_SYSTEM_GUIDE.md` | SciServer container、Job、persistent、scratch、ceph 和操作系统结构说明 |
| `Policies – SciServer.pdf` | 项目依据的平台存储与生命周期规则原文 |

项目不保留 Windows 启动器、`givernylocal`、keyring、本地路径、本地下载器、结果分卷器、`tar.zst` 归档或旧版全域派生数据格式。

## 4. 验证、Review 与安全性检测

### 4.1 自动化测试

离线测试命令：

```bash
python -m unittest discover -s tests -v
```

当前测试集合覆盖认证、坐标、请求规划、断点续跑、存储、谱计算、完整处理、失败提交和 GUI。测试必须使用小型 synthetic fixture 或 fake JHTDB backend；普通测试和安装不得触发完整真实数据请求。

### 4.2 输入验证门槛

输入进入 `validated` 状态前必须满足：

- 512 个 `128^3` tile 完整覆盖 `1024^3`，没有缺块或重叠；
- 每个 tile 写入后回读并与内存数据计算一致的 SHA-256；
- Giverny `(z,y,x,component)` 明确转换为 `(component,z,y,x)`；
- API 一基闭区间与本地零基半开切片映射正确；
- shape 为 `(3,1024,1024,1024)`、dtype 为 little-endian `float32`；
- 所有数值有限，不包含 NaN 或 Inf；
- 周期 wrap seam 与内部相邻面统计一致；
- manifest hash 与 Zarr attrs、catalog 状态一致。

### 4.3 物理与数值 Review

计算结果必须满足：

- 所有 FFT、导数和滤波都在完整 `1024^3` 周期域完成；
- 禁止“先裁剪、后 FFT”；
- 谱导数的数组轴与物理 `x/y/z` 方向映射明确；
- 流式 slab 滤波与内存参考实现数值一致；
- 常量场经过高斯滤波保持不变；
- 中心 raw/filtered 字段使用完全相同的空间坐标；
- 原始速度和滤波速度各自的全域相对无散度 RMS 不超过 `1e-4`；
- 原始速度和滤波速度各自的全域相对最大散度不超过 `1e-3`；
- 全域能量等式残差相对四场联合 RMS 不超过 `1e-4`；
- `|ΣS̄|/|ΣΠ|` 不超过 `1e-2`；不使用 `Σ|S̄|` 绝对归一化判据；
- regime 阈值和 occupancy 写入 QA。

`divergence.json` 和 `qa.json` 的 `divergence` 对象分别在 `unfiltered` 与 `filtered` 中记录两套全域统计。`s_bar_qa.json` 记录四个全域净和、各场平方和/RMS、相对能量等式残差、绝对诊断残差以及 `|ΣS̄|/|ΣΠ|`；不再计算 `Σ|S̄|` 绝对归一化 QA。只调整两个 S̄ 阈值时，程序会用报告中的缓存统计直接刷新 pass 和 metadata，不重扫四个全域数组。`s_bar_global_totals.html` 是可离线打开的四柱图。`cq.json` 和 `cq.html` 记录 Q1–Q4 的 stored/LES 贡献、条件均值、占比和四项 closure；`weak_asymmetry.json` 和 `weak_asymmetry.html` 记录全域 Π 正负分拆及净通量/RMS 指标。新计算在 Cq 扫描中同步累加 weak asymmetry；已有有效 Cq 报告但缺 weak 报告时只读取 `pi`，不会重扫两个 work 场。`qa-sbar`、`compute-cq` 与 `compute-weak-asymmetry` 只接受当前全域结果 schema。`process-center` 在任一速度场散度验证失败时保留失败 staging，不创建 `COMPLETE`；QA 失败会保留全域数据供诊断，并在 attrs/QA 中明确标记失败。

### 4.4 输出 Review 与提交规则

`finalize-result` 对每个字段执行：

- 字段存在性、rank、shape 和 dtype 检查；
- 分 chunk 有限值扫描；
- 最小值、最大值、字节数和完整 SHA-256 计算；
- `manifest.json`、`qa.json` 和 `divergence.json` 一致性记录；
- `s_bar_qa.json` 与 `s_bar_global_totals.html`；
- `cq.json` 与 `cq.html`；
- `weak_asymmetry.json` 与 `weak_asymmetry.html`；
- staging 到正式目录的同文件系统原子 rename；
- 最后创建包含 manifest hash 的 `COMPLETE`。

当前 schema 的正式目录已存在时拒绝覆盖。v4 可从已有全域 work 字段原子补齐全域 regime；v2/v3 必须补算缺失的中心外区域，不能直接改版本号。旧结果只会在 v5 staging 完成、已有中心四场重叠校验一致且字段校验通过后被替换。GUI 要求目录具有 `COMPLETE` 且 Zarr attrs 状态为 `complete`，不会把部分写入结果当成正式数据。

Review 正式结果时执行：

```bash
python -m jhtdb_pipeline status --config configs/pipeline.yaml
```

随后检查对应结果目录中的：

```text
COMPLETE
manifest.json
qa.json
divergence.json
s_bar_qa.json
s_bar_global_totals.html
cq.json
cq.html
center_result_sigma_<sigma_tag>.zarr/
```

### 4.5 安全性检测

- token 文件必须位于项目目录之外，权限不得宽于 `0600`；
- token 缺失、为空或权限过宽时 fail closed；
- shell 脚本使用 `set -euo pipefail` 和 `set +x`，避免失败后继续或打印 secret；
- token 不进入命令行参数、日志、manifest、QA、Zarr attrs 或 GUI；
- 配置加载器拒绝未知字段和不符合生产约束的 shape/path/阈值；
- 处理锁防止同一计算工作区被并发写入；
- 删除临时工作区前验证目标位于预期 scratch 父目录；
- persistent 提交不覆盖当前 schema 的正式结果；v4 regime 快速补齐以及 v2/v3 全场补算都只在校验完成后提交；
- GUI 以只读模式打开 Zarr，不提供上传、编辑或删除操作；
- `doctor` 在正式任务前检查 persistent/scratch 写权限、空间余量和 scratch 到期状态。

### 4.6 提交任务前的人工 Review 清单

1. Compute Image 是 Essentials 4.0，三个必需挂载卷都存在；
2. Quotas 页面仍有足够 persistent 余量；
3. `doctor` 的总状态为 `ok`；
4. 全部离线测试通过；
5. `plan` 显示预期的 8 个 request、512 个 checksum tile 和正确裁剪；
6. `smoke` 返回有限的 `(3,8,8,8)` `float32` 数据；
7. `time-index` 和 `sigma-grid` 与本次实验记录一致；
8. 没有另一个进程正在处理同一帧和尺度；
9. 任务结束后确认 `status`、`COMPLETE`、manifest 和 QA，再在 GUI 中 review。

## 5. 数据生命周期与正式输出

### 5.1 scratch 中间量

单帧 scratch 目录：

```text
JHTDB_RUNS/t000001/
├── velocity_cache.zarr/      # 完整 1024^3 × 3 速度，约 12 GiB 未压缩
└── work_sigma_1/             # FFT memmap 与工作缓冲，成功后自动清理
```

资源计划按完整速度缓存约 12 GiB、workspace 约 40 GiB、scratch 峰值约 52 GiB 估算。新增的 12 GiB 是三个全域 SGS transport 分量，用于直接计算 `s_bar` 的散度。scratch 是可重建数据，不是正式结果记录。

### 5.2 persistent 正式结果

结果路径示例：

```text
results/t000001_sigma_1/
├── center_result_sigma_1.zarr/
├── manifest.json
├── qa.json
├── divergence.json
├── s_bar_qa.json
├── s_bar_global_totals.html
├── cq.json
├── cq.html
├── weak_asymmetry.json
├── weak_asymmetry.html
└── COMPLETE
```

字段：

| 字段 | shape | dtype | 未压缩大小 |
|---|---:|---:|---:|
| `velocity` | `(3,512,512,512)` | `float32` | 1.5 GiB |
| `gradient` | `(3,3,512,512,512)` | `float32` | 4.5 GiB |
| `velocity_bar` | `(3,512,512,512)` | `float32` | 1.5 GiB |
| `gradient_bar` | `(3,3,512,512,512)` | `float32` | 4.5 GiB |
| `work_full` | `(1024,1024,1024)` | `float32` | 4 GiB |
| `work_resolved` | `(1024,1024,1024)` | `float32` | 4 GiB |
| `pi` | `(1024,1024,1024)` | `float32` | 4 GiB |
| `s_bar` | `(1024,1024,1024)` | `float32` | 4 GiB |
| `regime` | `(1024,1024,1024)` | `uint8` | 1 GiB |

单尺度未压缩合计约 29 GiB，另有 Zarr metadata、HTML 和文件系统开销。正式结果不打包、不分卷、不自动下载。

## 6. 失败恢复与当前边界

| 情况 | 行为与处理 |
|---|---|
| JHTDB request 失败 | 按配置重试；相同 `cache` 命令续跑，保留已验证 tile |
| 大 request 部分 tile 缺失 | 重新请求对应 `512^3` 大块，不生成第二份完整速度场 |
| Interactive 浏览器关闭 | 已提交 Compute Job 继续运行 |
| scratch run 超过配置生命周期 | `doctor` 拒绝继续信任，重新建立该帧缓存 |
| `process-center` 中断 | 同参数重跑；有效的 filtered velocity checkpoint 会复用，正式结果不受影响 |
| v2/v3 需要全域四场 | 运行 `backfill-full-fields`；复用 filtered workspace 或 raw cache，不自动 fetch |
| v4 需要全域 regime | 运行 `backfill-full-regime`；只读 persistent 全域 work 字段，不需要 temporary 或 FFT |
| `qa-sbar` 失败 | 保留全域数据和报告，检查 `s_bar_qa.json`，命令返回非零 |
| `compute-cq` 失败 | 保留正式字段，检查四项 closure、全域 work/pi 有限值和 `cq.json`，命令返回非零 |
| `compute-weak-asymmetry` 失败 | 保留正式字段，检查正负 Π closure、全域 pi 有限值和 `weak_asymmetry.json`，命令返回非零 |
| 散度失败 | staging 保留诊断数据，不创建 `COMPLETE` |
| `finalize-result` 前失败 | GUI 不显示 staging |
| persistent 空间不足 | 停止新任务并核对 Quotas，不覆盖或静默删除正式结果 |
| Streamlit 端口不可访问 | 增加服务器端 Jupyter viewer，不回退到本地数据流程 |

当前边界：

- 只支持 `isotropic1024coarse`、`velocity`、完整 `1024^3` 输入；
- 只支持单帧、单尺度正式任务；
- JHTDB 同一时刻严格限制为一个请求在途；
- 不实现本地下载、归档、结果分卷或本地 GUI；
- 批量尺度当前共享输入速度缓存，但每个尺度仍独立计算并保存完整结果；后续可进一步共享未滤波公共字段以减少计算和 persistent 占用。

## 7. 官方资料

- JHTDB Giverny：<https://github.com/sciserver/giverny>
- SciServer Compute：<https://apps.sciserver.org/compute/>
- SciServer Python/Jobs API：<https://www.sciserver.org/docs/sciscript-python/SciServer.html>
