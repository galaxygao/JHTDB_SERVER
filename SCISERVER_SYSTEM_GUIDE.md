# SciServer 系统结构与项目运行手册

本文只说明 SciServer 上的系统结构、各部分职责、系统配置和实际运行方式。当前实现是纯服务器流程：JHTDB 输入、计算、中间数据、正式结果和可视化均留在 SciServer，不与本地电脑交换科学数据。

## 1. 当前实现状态

单帧、多滤波尺度顺序批处理流程已经实现。当前入口仍使用 `single-frame`；它读取配置中的 sigma 列表，不存在单独的 `multi-scale` 命令。

2026-08-28，账户 `gaoxingqun` 的 Quotas 页面显示：

```text
Storage on FileServiceJHU: 100 GB
```

因此本项目按该账户的 100 GB persistent 配额设计。旧文档中的 10 GB 是平台旧默认值或不适用于该账户。项目没有可依赖的账户配额 API：`doctor` 能实测挂载文件系统的剩余空间，并显示配置中记录的 100 GB，但提交正式任务前仍应以 Quotas 页面为最终依据。

## 2. 系统总览

```text
浏览器
├── SciServer Compute 页面
│   ├── 创建 Interactive Compute container
│   └── 提交 shell-command Compute Job
│
└── Interactive container
    ├── JupyterLab / Linux bash Terminal
    ├── 环境准备、测试、doctor、plan、smoke
    └── 只读 Streamlit GUI

SciServer 挂载卷
├── Turbulence (ceph)：JHTDB 数据访问
├── persistent：代码、环境、状态、QA、manifest、正式结果
└── scratch：完整速度缓存和 FFT/memmap 工作区
```

Interactive container 和 Compute Job 是两个计算入口，但可挂载同一组 persistent、scratch 和 Turbulence 卷。容器自身文件系统不是正式存储位置；跨容器或跨 job 保留的内容必须位于挂载卷。

## 3. 各部分的特点和功能

### 3.1 Interactive Compute container

交互容器由 Compute 页面创建，适合：

- 打开 JupyterLab 和 Linux `bash` Terminal；
- 在 persistent 中建立项目环境；
- 运行离线测试、`doctor`、`plan` 和小型真实 `smoke`；
- 查看 job 状态与结果；
- 启动只读服务器 GUI。

它不适合承载正式长计算。关闭浏览器不会停止已经提交的 Compute Job，但停止或删除交互容器会终止其中的 Terminal 和 GUI 进程。

### 3.2 Terminal 与 Windows CMD 的区别

SciServer 有命令行，但通常是 Linux `bash`，不是 Windows `cmd.exe` 或 PowerShell。路径使用 `/`，环境激活使用：

```bash
source .venv/bin/activate
```

项目不保留 `.bat`、Windows 盘符、Windows Credential Manager 或本地路径。

### 3.3 Compute Image

首版使用：

```text
SciServer Essentials 4.0
```

镜像提供基础 Python/Jupyter 环境。项目用 persistent 中的 `.venv` 补充自己的包，因此交互容器和 job 必须选择兼容的 image。首版不需要自行创建 Docker image。

`SciServer Essentials 6.0` 不适用于当前 `isotropic1024coarse` 流程。该旧数据集仍走 legacy `pyJHTDB` 后端；Essentials 6.0/Python 3.12 会安装已经弃用、导入时主动报错的 `pyJHTDB` 占位包。必须选择自带可用 legacy runtime 的 Essentials 4.0（Python 3.9），保留镜像自带的 Giverny/pyJHTDB，且不能跨镜像复用 6.0 创建的 `.venv`。

### 3.4 Compute Job

正式计算使用 shell-command Compute Job。特点是：

- 提交后在后台运行，不依赖浏览器保持打开；
- 每次提交都必须明确选择 image 和挂载卷；
- 可以读取 persistent、scratch 和 Turbulence；
- 有明确的成功或失败状态，适合保存日志和复跑；
- 不用于长期承载 GUI。

### 3.5 Turbulence (ceph)

这是 SciServer 的 JHTDB 数据与 metadata 挂载。项目使用原生 `giverny`；但 `isotropic1024coarse` 属于其 legacy pyJHTDB 路径，并非直接读取新式 Ceph/Zarr 数据。项目因此以 8 个 `512^3` request block 严格串行读取 velocity，避免对 legacy gSOAP 重复执行 512 次小请求。

每个 request 在内存中拆成 64 个 `128^3` checksum tile，逐块填入 scratch 中预分配的完整速度 Zarr 数组并做回读 SHA-256。大请求减少 legacy pyJHTDB/gSOAP 初始化和网络往返；小存储块保留细粒度完整性检查。不会先生成待拼接的分卷包，也不会建立第二份完整速度副本。

### 3.6 persistent

账户路径：

```text
/home/idies/workspace/Storage/gaoxingqun/persistent
```

项目使用：

```text
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
```

persistent 用于：

- 项目代码和 `.venv`；
- SQLite catalog、输入 manifest、QA、锁和日志；
- `results/.staging` 中尚未正式提交的中心结果；
- `results/<result_id>` 中带 `COMPLETE` 的权威结果。

persistent 不保存完整 `1024^3` 速度缓存或全域 FFT 工作场。单尺度中心正式结果未压缩约 14.125 GiB，配置另保留至少 15 GiB 安全余量。

### 3.7 scratch / Temporary

项目路径：

```text
/home/idies/workspace/Temporary/gaoxingqun/scratch/JHTDB_RUNS
```

scratch 保存：

- `velocity_cache.zarr`：完整 `1024^3 × 3` 速度缓存，约 12 GiB 未压缩；
- 全域滤波速度和谱运算 memmap；
- 可重建的临时工作量。

按随项目保存的 SciServer policy PDF，Temporary 数据没有备份，并可能在创建约 72 小时后被删除；修改文件或复跑不能被视为重新获得 72 小时。任何 scratch 内容都必须按“可能消失”处理。

当前资源计划约为：

| 内容 | 未压缩规模 |
|---|---:|
| 完整速度缓存 | 12 GiB |
| 谱计算 workspace | 40 GiB |
| scratch 峰值 | 约 52 GiB，另加 16 GiB 安全余量 |

### 3.8 persistent 正式结果

结果目录：

```text
results/
├── .staging/<result_id>/
└── <result_id>/
    ├── center_result_sigma_<sigma_tag>.zarr/
    ├── divergence.json
    ├── qa.json
    ├── manifest.json
    └── COMPLETE
```

`COMPLETE` 最后创建。GUI 只列出有该标记的目录；中断或校验失败的 `.staging` 不会冒充正式结果。

## 4. 科学处理流程

```text
JHTDB velocity
  → 8 个 512^3 request block 严格串行获取
  → 拆成 512 个 128^3 tile 填入 scratch 全域速度数组
  → tile 覆盖、SHA-256 回读、shape/dtype/有限值验证
  → 在完整 1024^3 周期域上做谱导数和谱高斯滤波
  → 原始速度与滤波速度的全域无散度 QA
  → 裁剪 x,y,z = [256,768)
  → 写入 persistent staging
  → 输出字段逐数组校验和散列
  → staging 原子改名为正式目录
  → 最后写 COMPLETE
```

绝不能先裁剪再 FFT。正式 Zarr 包含：

- `velocity`：未滤波中心速度；
- `gradient`：未滤波中心速度梯度；
- `velocity_bar`：滤波中心速度；
- `gradient_bar`：滤波中心速度梯度；
- `work_full`、`work_resolved`；
- `pi = tau_ij ∂_j velocity_bar_i`；
- `s_bar = ∂_j(velocity_bar_i tau_ij)`；
- `regime`。

JHTDB request block、checksum tile 和 Zarr chunk 都是服务器内部的读写单位，不是下载分卷。项目不生成归档包或传输分卷。

## 5. 平台配置

创建交互容器和正式 job 时都选择：

| 项目 | 设置 |
|---|---|
| Compute Image | `SciServer Essentials 4.0` |
| Data Volume | `Turbulence (ceph)` |
| User Volume | `persistent`，读写 |
| User Volume | `scratch`，读写 |

项目实际配置位于 `configs/pipeline.yaml`，已经写入 `gaoxingqun` 的服务器路径。重要参数包括：

```yaml
platform:
  state_root: /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/state
  run_root: /home/idies/workspace/Temporary/gaoxingqun/scratch/JHTDB_RUNS
  result_root: /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/results
  scratch_retention_hours: 72

auth:
  token_file: /home/idies/workspace/Storage/gaoxingqun/persistent/.secrets/jhtdb_token

jhtdb:
  request_shape: [512, 512, 512]
  tile_shape: [128, 128, 128]

storage:
  compression_level: 3
  compression_threads: 8
  persistent_capacity_gb_observed: 100
  persistent_safety_reserve_gib: 15
  scratch_safety_reserve_gib: 16

physics:
  sigma_grid: [1.0, 2.0, 3.0]
  crop_start: [256, 256, 256]
  crop_shape: [512, 512, 512]
  fft_workers: 16
  fft_slab_width: 32
```

配置解析器拒绝未知字段，避免拼写错误被静默忽略。`persistent_capacity_gb_observed` 是 Quotas 页面观测值，不代表程序能通过文件系统调用读取账户硬配额。

JHTDB 请求仍严格串行；`fft_workers` 只并行容器内的 FFT。`fft_slab_width: 32` 使单个 FFT 输入块约为 128 MiB，配合 16 个 FFT workers 和 8 个 Blosc 压缩线程使用多核资源，同时继续把约 40 GiB 中间场存入 scratch memmap。

参数调整规则：

- `fft_workers` 不超过 Job 分配的 CPU 核数；可按 8、16、32 逐档实测，线程更多不保证更快；
- `fft_slab_width=16/32/64` 的单输入块约为 64/128/256 MiB，调大通常减少批次，但 FFT 实际瞬时内存是输入块的数倍；
- `compression_level=1` 偏速度，`3` 偏平衡；`compression_threads` 通常设为 CPU 核数的 1/4–1/2；
- sigma 数量使计算时间和 persistent 结果量近似线性增长，但顺序批处理不增加单尺度 scratch 峰值；
- `cleanup_scratch_on_success` 在批量任务中必须保持 `true`，否则每个尺度约 40 GiB workspace 会累积；
- safety reserve 和 `persistent_capacity_gb_observed` 不会加速；生产 crop、request 和 tile shape 不作为性能参数修改。

修改 YAML 后运行 `plan` 核对资源。正式脚本通过 `tee` 持久化日志，Rich 进度可能只在阶段结束时显示最终 `42/42`；Interactive Terminal 需要实时进度时可直接运行 `python -m jhtdb_pipeline single-frame --time-index 1 --config configs/pipeline.yaml`。两种入口的物理计算相同，不要在已有任务运行时重复启动。

## 6. 首次建立环境

先在 Compute 页面创建一个交互容器，挂载上一节的三个卷。打开 JupyterLab Terminal 后运行：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
bash scripts/bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` 只建立服务器环境和检查依赖，不会访问完整 JHTDB 数据。项目和 `.venv` 都在 persistent，因此新容器和 job 可以继续使用。

Essentials 4.0 的基础环境含 TensorFlow、Dash 等旧包；项目虚拟环境又必须通过 `--system-site-packages` 继承 legacy pyJHTDB。安装时可能出现这些无关包的 dependency-conflict 警告。项目不使用它们，`bootstrap.sh` 以所需模块和 JHTDB runtime 的实际导入作为成功条件，不对整个镜像执行 `pip check`。

## 7. 配置 JHTDB token

token 不得写入 YAML、Git、Notebook、CLI 参数、日志、QA、manifest 或 GUI。建议在交互 Terminal 中写入项目目录之外的受保护文件：

```bash
SECRET_ROOT=/home/idies/workspace/Storage/gaoxingqun/persistent/.secrets
mkdir -p "$SECRET_ROOT"
chmod 700 "$SECRET_ROOT"
read -rsp "JHTDB token: " JHTDB_TOKEN_INPUT
echo
(umask 077; printf '%s' "$JHTDB_TOKEN_INPUT" > "$SECRET_ROOT/jhtdb_token")
unset JHTDB_TOKEN_INPUT
```

验证时只报告来源，不显示内容：

```bash
python -m jhtdb_pipeline auth status --config configs/pipeline.yaml
```

程序也接受当前进程中的 `JHTDB_TOKEN` 环境变量，但 Compute Job 通常不会自动继承交互 Terminal 的临时环境，因此受保护文件更适合当前流程。

## 8. 正式运行前检查

在交互容器中依次运行：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
source .venv/bin/activate

python -m jhtdb_pipeline doctor --config configs/pipeline.yaml
python -m unittest discover -s tests -v
python -m jhtdb_pipeline plan --time-index 1 --config configs/pipeline.yaml
python -m jhtdb_pipeline smoke --time-index 1 --config configs/pipeline.yaml
```

| 命令 | 是否访问 JHTDB | 功能 |
|---|---:|---|
| `doctor` | 否 | 检查 Linux、挂载路径、写权限、依赖、token、文件系统空间和已存在 run 的到期时间 |
| `unittest` | 否 | 运行小网格离线单元与端到端测试 |
| `plan` | 否 | 显示 tile 数量、请求策略和存储峰值 |
| `smoke` | 是 | 读取一个 `8^3` 真实块，检查 token、Giverny、轴顺序和 dtype |

`smoke` 会在 scratch 中创建一个很小的 Giverny 输出目录，但不会开始完整 1024³ 获取。

## 9. 提交第一帧 Compute Job

在 Jobs 页面创建 shell-command job，并选择与交互容器兼容的 domain、同一个 image、Turbulence、可写 persistent 和可写 scratch。

命令为：

```bash
bash -lc 'cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA && source .venv/bin/activate && bash scripts/run_stage.sh single-frame --time-index 1'
```

脚本会执行：

```text
doctor → cache/validate-input → process-center → finalize-result
```

运行日志写入：

```text
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/state/logs
```

断点行为：

- 已验证的 `128^3` tile 会回读 SHA-256 后跳过；某个 `512^3` request 仍有缺块时只重新请求该大块；
- 谱处理若中断，会清理同一 result 的旧 staging/workspace 并从处理阶段开头重算；
- 已存在且带 `COMPLETE` 的相同 `time_index + sigma_grid` 当前结果不会被覆盖；缺少 `pi/s_bar` 的旧结果会在新版 staging 完成并通过校验后被替换；
- 失败的 staging 不会出现在 GUI 正式结果列表中。

查看状态：

```bash
python -m jhtdb_pipeline status --config configs/pipeline.yaml
```

## 10. 分阶段命令

正常正式运行只需要 `single-frame`。排错时可手动执行：

| 命令 | 功能 |
|---|---|
| `auth status` | 只报告 token 是否可用和来源 |
| `doctor` | 平台与存储前检 |
| `plan --time-index N` | 资源和请求计划 |
| `smoke --time-index N` | 小块真实读取 |
| `cache --time-index N` | 串行构建或续跑完整速度缓存 |
| `validate-input --time-index N` | 重新验证完整输入缓存 |
| `process-center --time-index N --sigma-grid S` | 全域谱处理并写 persistent staging |
| `finalize-result --time-index N --sigma-grid S` | 校验 staging 并提交正式结果 |
| `upgrade-result --time-index N --sigma-grid S` | 用已有中心 `gradient_bar` 检查滤波散度并将完整 v2 结果升级为 v3，无需重算 |
| `single-frame --time-index N [--sigma-grid S]` | 串联完整流程；默认批量处理配置列表，指定参数时只跑一个尺度 |
| `status` | 列出输入和正式/非正式结果状态 |
| `gui` | 启动只读服务器 GUI |

## 11. 在服务器上运行 GUI

GUI 在交互容器中运行，不在 Compute Job 中运行：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
source .venv/bin/activate
python -m jhtdb_pipeline gui --config configs/pipeline.yaml --port 8501
```

它监听 `0.0.0.0:8501`，只读取 `persistent/results` 中带 `COMPLETE` 的正式 Zarr，并按需载入二维切片。支持 raw/filtered velocity、raw/filtered gradient、work、`pi`、`s_bar`、regime、QA 和 manifest。

需要通过当前 SciServer compute domain 提供的端口入口打开。若该 domain 不提供 Streamlit 端口代理，当前 CLI 不会自动建立 Jupyter 内嵌 viewer；应先补充服务器端 viewer，而不是把数据下载到本地。

## 12. 失败与恢复

| 情况 | 处理 |
|---|---|
| 浏览器关闭 | 已提交 job 继续运行；之后回 Jobs 页面查看 |
| 交互容器停止 | 重新启动容器；persistent 和未过期 scratch 仍按平台规则存在 |
| request 请求失败 | 相同命令复跑；已验证 tile 会跳过，只重取含缺块的 request |
| scratch run 接近或超过 72 小时 | 不再信任缓存；清理该 scratch run 后重新获取 |
| 谱处理或 staging 中断 | 相同参数复跑；处理阶段从头重算，正式结果不受影响 |
| persistent 空间不足 | 停止新结果写入，先核对 Quotas；不要删除正式结果来掩盖容量规划错误 |
| `COMPLETE` 不存在 | 该目录不是正式结果，GUI 不读取 |
| Streamlit 无法打开 | 核对端口代理；后续增加服务器端 Jupyter viewer |

## 13. 当前边界

- 当前一个正式 job 处理一帧，并顺序计算配置中的多个 `sigma_grid`；
- 多个尺度共享输入速度缓存，但当前不共享正式结果中的公共字段；
- 当前不实现本地下载、归档、传输分卷或本地 GUI；
- 当前不创建自定义 container image；
- 真实 `1024^3` 首帧尚需在 SciServer 上完成 `doctor`、离线测试和 `smoke` 后再提交。

## 14. 随项目保留的规则与官方入口

- 存储规则：[`Policies – SciServer.pdf`](Policies%20%E2%80%93%20SciServer.pdf)
- JHTDB Giverny：<https://github.com/sciserver/giverny>
- SciServer Compute：<https://apps.sciserver.org/compute/>
- SciServer Python/Jobs API：<https://www.sciserver.org/docs/sciscript-python/SciServer.html>
