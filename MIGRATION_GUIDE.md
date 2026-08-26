# 从局部切片项目迁移到全周期域速度流水线

> 状态：**迁移实施中**。旧局部数据和旧实现已从当前工作树删除；新下载、catalog、Zarr、验证、GUI 和周期物理模块已经生成并通过离线测试。在线 smoke 与三个全域快照等待本地凭据配置。

## 1. 迁移原则

1. 原始事实层只保存 JHTDB 返回的全周期域 `float32` 速度。
2. 下载、数据验收和物理计算是三个独立阶段，各自有状态、checksum 和明确输入输出。
3. 空间和时间都使用整数索引作为身份；浮点坐标只作物理坐标，不作去重键。
4. 一个 canonical raw store；不得同时长期保存 tile 文件、拼接副本和另一种 raw 格式。
5. 自动完整性验证通过后，物理模块即可读取该快照；GUI 仅用于只读观察，不是门禁。
6. 任何失败都应可恢复；任何重跑都不得产生重复数据。
7. 准确性优先：不静默插值、不降精度、不量化、不用局部 block 冒充周期域。
8. 所有 JHTDB 网络查询全局严格串行；CLI 不暴露并发选项。
9. 单次任务只接受一个明确的 `time_index`；catalog 可管理多次独立任务，但不自动抓取时间范围。

## 2. 当前实现与目标实现的映射

| 当前实现 | 迁移后 | 原因 |
|---|---|---|
| `getData` 点查询 | `getCutout` 原始网格 tile | 全域下载且不做插值 |
| `16³` block | `[1,1024]³` 唯一完整覆盖 | 使用真实周期域 |
| block start/shape | 固定全域 + tile plan | tile 只是传输单位，不是物理域 |
| halo/core | 删除 | 全域周期算法不裁 core |
| 下载 velocity + FD8 + FD6 | 只下载 velocity | 梯度本地从全域速度计算 |
| 局部 valid Gaussian | 周期谱 Gaussian | 正确跨越周期边界 |
| 数据库局部 FD 梯度 | 全域周期谱导数 | 与周期 DNS 更一致 |
| 单个 `.npz` raw | chunked Zarr + SQLite catalog | 适合 12 GiB/快照、断点续传和局部访问 |
| 局部块 dashboard | 全域 coverage/slice/seam/diagnostics GUI | 可视化检查完整数据 |
| FD6/FD8 disagreement | 解析场、谱性质、散度和 checksum QA | audit 对象已改变 |

## 3. 审阅批准后的删除范围

### 3.1 旧数据

删除当前工作树中的：

```text
data/cache/**
data/raw/**
data/derived/**
data/reports/**
```

这些内容是旧 `16³` 局部块任务的下载、梯度、派生和报告，不能进入新全域 catalog。删除前先输出精确文件清单和大小；删除后提交一个可审计的迁移 commit。

注意：它们已存在于当前 Git 历史。普通 `git rm` 只从新版本删除，旧 commit 仍可恢复。若审阅选择“彻底清除 GitHub 历史”，需要单独执行历史重写、验证新仓库大小并 force-push；该操作具有破坏性，不与普通代码重构混在同一步执行。

### 3.2 旧源码与入口

已经删除或整体替换：

- `src/jhtdb_regimes/grid.py` 中 block point grid、4000 点 batch 和局部 reshape；
- `src/jhtdb_regimes/jhtdb_client.py` 中 `getData`、gradient column、FD6/FD8、testing-token fallback 和旧 `.npz` cache；
- `src/jhtdb_regimes/verify.py` 中局部 FD6/FD8、halo/core 和局部 direct-filter 审计；
- `src/jhtdb_regimes/dashboard.py` 中面向局部 raw/derived 的旧页面；
- `configs/task0.yaml` 中 block、halo、gradient primary/audit、testing-token 和旧路径字段；
- `task0.py`、`task0.bat`、`run_task0.ps1`、`dashboard.py`、`dashboard.bat` 的旧命令入口；
- 针对上述旧行为的 tests；
- `task 0.md` 的旧局部切片技术方案（旧 README 按用户要求保留为 `old readme.md`）。

`physics.py` 不直接保留旧实现：保留公式和分类语义，但以全周期域、流式谱算法重写。

## 4. 新实现阶段

### 阶段 A：配置、metadata 与计划器

- 新建强类型配置 schema；
- 新增 `auth set/status/delete` CLI：隐藏输入，并通过系统 keyring 保存到当前 Windows 用户的 Credential Manager；
- 下载器只允许从系统凭据库取得 token，不从环境变量、YAML、命令参数或项目文件读取；
- 运行时读取并缓存官方 metadata 摘要；
- 用 metadata 验证 dataset、网格、周期性和可用时间索引；
- 要求每次任务恰好一个 `time_index`，拒绝列表、范围和隐式循环；
- 计算 backend 限额、tile 数、请求数、raw/derived/scratch 最坏空间与安全余量；
- 在 full-domain 计划中显示 JHTDB intended-usage 警告；
- 数据根目录若位于仓库或云同步目录内则拒绝正式下载；
- 空间不足时在第一次网络请求前失败。

验收：`plan` 完全离线可测试；metadata 变化会改变 dataset identity；token 不进入输出。

### 阶段 B：catalog 与 raw store

- 建立 SQLite schema、唯一约束、状态迁移和单写者锁；
- 建立 Zarr schema、坐标、attrs 和 lossless codec；
- tile 写入目标切片，回读后生成 checksum；
- crash 后未完成 tile 可安全覆盖；
- 完成的 tile 重跑只校验并跳过。

验收：合成小网格上的故意中断、重复运行、checksum 损坏和双写者测试全部通过。

### 阶段 C：JHTDB downloader 与终端进度

- 正式网格下载封装 `GetCutout`；本地 backend 遵守 3 GB 单次限额，SciServer backend 遵守 16 GB 单次限额；
- `GetData` 仅用于少量随机点独立复核，受 200 万点上限约束但本项目远低于该上限；
- 只允许 `velocity`；
- 规范化 API 返回的 axis/component/time；
- 使用全局单请求执行器，不创建并发 JHTDB 请求；
- retry/backoff 不吞掉最终错误；
- 进度显示 snapshot、tile、吞吐、ETA、重试和空间；
- smoke 只请求很小 tile 和一个时间索引。

验收：在线 smoke 与独立抽样一致；日志全文搜索不到 token；模拟超时后恢复正确。

### 阶段 D：自动验证与 GUI

- 覆盖、唯一性、shape、dtype、有限值、时间和坐标检查；
- tile seam 与 periodic wrap seam 检查；
- 生成全域统计和可视化金字塔/缓存，缓存可随时重建；
- Streamlit GUI 只读 raw，不写 accept/reject 或人工审批状态；
- raw hash 变化后自动刷新统计和可视化缓存。

验收：缺 tile、重 tile、错轴、错分量、NaN、重复时间和伪重复端面都能被测试捕获。

### 阶段 E：周期物理计算

- 解析周期场测试先行；
- 实现逐轴/分 slab 的谱导数；
- 实现可分离周期谱 Gaussian；
- 重建 acceleration、work 和 regime；
- 每次只处理一个已通过自动验证的快照；
- scratch 有状态、有空间预检，成功后清理；
- 最终 derived 保存输入和算法 provenance。

验收：解析导数误差、周期 seam、滤波 transfer function、轴顺序、常数场、单模 Fourier 场、Parseval/能量、散度和 regime 边界测试通过。

### 阶段 F：清理与文档收口

- 删除所有旧功能和失效测试；
- 更新 `.gitignore`，确保大型数据永不再次提交；
- 新 README 中的命令逐条 smoke test；
- 生成迁移清单：删除、保留、新增、数据兼容性和已知限制；
- 提交代码，但正式全域下载作为单独、显式操作，不因安装或测试自动触发。

## 5. 建议的新配置形状

当前可运行配置 schema：

```yaml
dataset: isotropic1024coarse
variable: velocity

auth:
  backend: windows_credential_manager
  service: jhtdb_pipeline
  # 仅用于 local Windows；选择 SciServer 时必须单独确定安全凭据 backend

time:
  # 每次命令只允许一个索引；通常由 --time-index 显式传入
  index: 1

download:
  backend: local  # local 或 sciserver
  tile_shape: [128, 128, 128]
  retries: 5
  backoff_seconds: 2
  enforce_single_in_flight_query: true
  show_full_domain_usage_warning: true

storage:
  root: "待填写的仓库外绝对路径"
  raw_store: raw/velocity.zarr
  catalog: catalog.sqlite
  compression: lossless
  safety_free_space_gib: 20

validation:
  random_point_count: 64
  seam_width: 4

gui:
  read_only: true

physics:
  derivative: periodic_spectral
  filter: periodic_spectral_gaussian
  sigma_grid: 1.0
  epsilon_abs: 0.0
  epsilon_rel: 0.001
  keep_intermediates: false
```

时间索引示例 `1` 仅用于先完成一个 12 GiB 快照的端到端验收。配置和 CLI 均不接受自动批量时间范围；后续快照需要作为新的明确任务提交并由 catalog 去重排序。

## 6. 测试矩阵

| 层级 | 不访问网络 | 主要内容 |
|---|---:|---|
| unit | 是 | tile 计划、坐标、唯一键、状态机、hash、谱导数、滤波、regime |
| integration | 是 | 合成 Zarr、断点恢复、GUI helper、scratch 生命周期 |
| online smoke | 否 | 小 cutout、token、metadata、组件/轴/时间 |
| one-tile online | 否 | 真实 `128³` tile 写入与独立抽样 |
| one-snapshot end-to-end | 否 | 512 tile 全域覆盖、GUI 可视化、物理小规模/流式检查 |
| serial-request policy | 否 | 证明重试、复核和下载之间不存在重叠 JHTDB 请求 |

正式下载不会放入普通测试命令。

## 7. 数据准确性风险与控制

| 风险 | 控制 |
|---|---|
| 1-based/0-based 偏移 | 所有 API range 单测；规范 store 使用 0-based array index |
| x/y/z 置换 | 单调坐标场和独立点查询验证 |
| ux/uy/uz 置换 | 按命名解析并保存 component coordinate |
| tile 重叠/缺口 | 唯一键 + 覆盖位图 + 每点一次写入证明 |
| 重复周期端面 | store shape 固定 1024；禁止保存索引 1025 |
| 时间重复/乱序 | `time_index` 主键；最终坐标严格递增 |
| float 精度改变 | 输入和 raw 均为 float32；无损压缩；字节 checksum |
| 中断留下半块 | ledger 仅在回读通过后提交 verified；未提交块覆盖重写 |
| raw 被修改后仍计算 | manifest hash 与自动验证状态门禁 |
| 内存不足导致近似降级 | preflight 后失败；绝不静默更换算法 |

## 8. 审阅出口条件

只有以下事项均被用户确认后，才进入代码和删除阶段：

- 最终 `time_index` 计划；
- local 或 SciServer backend，以及完整 3D 场获取方式符合 JHTDB 使用建议；若选择 SciServer，还需确认其独立凭据 backend；
- 仓库外数据根目录和可用空间；
- 周期谱导数/滤波选择；
- 旧数据只从 HEAD 删除，还是连 Git 历史一起清除；
- README 对项目命令、数据 schema 和验收标准的描述得到批准。
