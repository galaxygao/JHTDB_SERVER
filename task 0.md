# Task 0：基于 JHTDB 网格速度与数据库差分梯度的多快照 regime 计算

> 依据：[test.md](test.md)  
> 版本：PC + JHTDB testing token 路线  
> 状态：脚本与 testing token API 已部署；8 点 × 2 时间在线 smoke test 已通过；尚未执行正式 100-snapshot 下载  
> 日期：2026-08-25

## 1. 修订后的目标

考虑当前只有 JHTDB testing token，并且本机约有 100 GB 磁盘、16 GB RAM、6 GB GPU 显存，Task 0 不再下载完整 (1024^3) snapshot，也不做完整域 Fourier 求导。

新方案直接在 JHTDB 的真实网格点上获取：

1. velocity field：`spatial_operator="field"`、`spatial_method="none"`；
2. velocity gradient：`spatial_operator="gradient"`、`spatial_method="fd8noint"`；
3. 多个真实存储时间的同一空间网格块；
4. 在本地计算原始 advective acceleration、局部滤波量、两种 filtered acceleration work 和 Q1–Q4 regimes；
5. 使用 `fd6noint` 审计部分数据，标记对导数方法敏感的 regime。

最终核心输出仍为

\[
\bar a_i=\overline{u_j\partial_j u_i},
\qquad
\overline{\bar a}_i
=\bar u_j\partial_j\bar u_i,
\]

\[
W_{\rm full}=\bar{\boldsymbol u}\cdot\bar{\boldsymbol a},
\qquad
W_{\rm res}
=\bar{\boldsymbol u}\cdot\overline{\bar{\boldsymbol a}},
\]

以及 Q1–Q4 regime。

## 2. 本方案的科学边界

### 2.1 可以完成的内容

- 使用 JHTDB 官方局部有限差分梯度；
- 在真实 DNS 网格上计算 \(u_j\partial_j u_i\)；
- 使用带 halo 的局部卷积进行滤波；
- 计算 \(\bar a\)、\(\overline{\bar a}\)、两个 work scalar 和 regimes；
- 用更多时间快照形成统计样本；
- 定量报告数据库梯度方法对 regime 的影响。

### 2.2 不再声称完成的内容

- 不声称使用了 DNS 的全局 spectral derivative；
- 不声称滤波后的场在 spectral sense 下严格无散；
- 不把局部空间块当作周期域做 FFT；
- 不声称严格复现 PDF 中尚未给出 transfer function 的 “Gaussian-smoothed sharp spectral filter”；
- 不计算正式的全域 SGS energy budget；
- 不把数据库 `fd8noint` 的结果称为无误差真值。

JHTDB 官方对 `isotropic1024` 的说明指出，数据库提供的局部 finite-difference/spline gradient 相对 DNS 全局谱导数约有 **7% RMS error**。该误差会传播到 acceleration 和 work，且可能改变接近零点的 regime。因此本 Task 必须输出 uncertainty/boundary 类别，不能强迫所有点进入 Q1–Q4。

## 3. 数据集与 API 参数

### 3.1 数据集

```yaml
dataset: isotropic1024coarse
variable: velocity
grid_shape: [1024, 1024, 1024]
domain: [0, 2*pi]
grid_spacing: 2*pi/1024
periodic: [true, true, true]
temporal_method: none
```

所有查询坐标必须精确落在网格上：

\[
x_i=i\frac{2\pi}{1024},
\quad
y_j=j\frac{2\pi}{1024},
\quad
z_k=k\frac{2\pi}{1024},
\]

其中 \(i,j,k\in\{0,\ldots,1023\}\)。坐标使用 float64 生成并按官方接口要求传递。

### 3.2 JHTDB 查询

velocity：

```python
getData(
    cube,
    var="velocity",
    timepoint_original=t_start,
    temporal_method="none",
    spatial_method_original="none",
    spatial_operator="field",
    points=points,
    option=[t_end, delta_t],
    return_times=True,
)
```

velocity gradient：

```python
getData(
    cube,
    var="velocity",
    timepoint_original=t_start,
    temporal_method="none",
    spatial_method_original="fd8noint",
    spatial_operator="gradient",
    points=points,
    option=[t_end, delta_t],
    return_times=True,
)
```

梯度分量必须根据返回 DataFrame 的列名映射到

\[
G_{ij}=\partial_j u_i,
\]

禁止未经验证地把 9 列直接 reshape。

## 4. Testing token 约束

### 4.1 点数与并发

JHTDB 官网说明 testing token 用于少于 4096 点的请求。为避免依赖“4096 是否包含端点”的解释，配置固定：

```yaml
max_points_per_query: 4000
max_concurrent_queries: 1
```

JHTDB 2025-06-13 公告要求 testing token 一次只提交一个查询，并等待该查询完成。因此程序必须串行查询，不使用线程池、异步并发或多进程并发访问服务。

### 4.2 token 管理

- 从环境变量 `JHTDB_TOKEN` 读取；
- 配置文件只写 `token_env: JHTDB_TOKEN`；
- token 不进入日志、manifest、异常快照或版本库；
- 若检测到 testing token，自动强制 `max_points_per_query <= 4000` 和 concurrency 1。

### 4.3 Time-series 请求

同一批空间点优先用 `getData` 的 time-series option 一次请求一段时间，而不是每个 snapshot 单独发请求。客户端仍需检查当前 Giverny metadata 对 `points × times` 的总限制，并把每段成功结果立即写入断点文件。

基线规模：

- 空间点：\(16^3=4096\)，拆为 4000 点和 96 点两批；
- 正式时间样本：100 个 snapshots；
- 每个时间分块：20 个 snapshots，共 5 段；
- velocity：\(2\times5=10\) 个串行请求；
- `fd8noint` gradient：\(2\times5=10\) 个串行请求；
- 基线合计：约 20 个串行请求；
- 完整 `fd6noint` 审计：再增加约 10 个串行请求。

每个请求结束并校验、落盘后才开始下一个。服务拒绝 time-series 或返回规模限制时，程序依次把时间分块从 20 缩小到 10、5、1，但保持目标时间列表不变；不得通过并发绕过限制。

## 5. 空间块、halo 与 core

### 5.1 基线布局

```yaml
block_shape: [16, 16, 16]
halo: [4, 4, 4]
core_shape: [8, 8, 8]
```

block 含 4096 个点；滤波后只保留中心 \(8^3=512\) 个 core 点。halo 中的数据只为局部 convolution 提供邻域，不进入最终统计。

100 个时间快照产生

\[
100\times8^3=51200
\]

个网格点级 regime 样本。

### 5.2 空间位置

基线选择远离数组表示边界的固定块，例如：

```yaml
block_start_ijk: [504, 504, 504]
block_shape: [16, 16, 16]
```

其物理位置由网格 index 精确换算。固定空间块便于比较时间变化。

后续若使用多个 block，应分别保存 `block_id`，并按 JHTDB 的全局周期性处理跨越 0/1024 边界的 index；不能把局部 block 自身设为周期。

### 5.3 halo 限制

halo=4 只允许 support radius 不超过 4 个网格点的滤波核。若滤波尺度需要更大 support，应扩大 block 并拆成更多小于 4000 点的串行 batch；不能在 block 边缘使用反射、最近值填充或局部 periodic wrap 来伪造邻域。

## 6. 时间快照设计

### 6.1 基线时间

```yaml
t_start: 0.0
t_end: 9.9
delta_t: 0.1
expected_snapshots: 100
time_chunk_size: 20
temporal_method: none
```

这些时间均为 coarse 存储间隔 0.002 的整数倍，并覆盖数据库约五个大涡周转时间。运行时必须读取 API 实际返回的时间，检查：

- 数量是否为 100；
- 是否与请求时间一致；
- 是否没有 `pchip`；
- 是否严格递增且无重复。

先用 2 个 snapshots 做在线 smoke test，再用 20 个完成 pilot，最后扩展至 100 个。若当前 metadata 或服务不接受单段 20 个时间，缩小请求分段但不改变 100 个目标时间；若某个目标时间最终失败，必须在 manifest 中明确记录缺失及错误，不能用相邻时间代替。

### 6.2 统计独立性

同一 block 内相邻网格点高度相关，时间上也可能相关。因此不能把 51200 个点全部当成独立样本计算置信区间。

统计报告应至少包含：

- 每个 snapshot 的 Q1–Q4 occupancy；
- across-snapshot mean 和标准差；
- 以 snapshot 为单位的 bootstrap；
- 各 \(W\) 的时间自相关或至少 lag-1 correlation；
- boundary/uncertain fraction。

## 7. 数据布局与存储

本方案数据规模很小，不需要 GPU，也不需要 Zarr 才能运行；推荐使用 xarray + NetCDF/HDF5 或 Zarr 保留命名维度。100 个 snapshots 的 velocity、FD8 gradient 和 FD6 gradient 即使用 float64，总原始数组量也低于 70 MB；加上派生量、断点和报告仍远低于本机 100 GB 存储限制。

统一布局：

```text
velocity(time, component, z, y, x)
velocity_gradient(time, velocity_component, derivative_axis, z, y, x)
```

其中：

```text
component = [ux, uy, uz]
derivative_axis = [x, y, z]
```

不得在无命名信息的 `(time, point, 9)` 数组上直接进行物理计算。下载后先重建规则网格并验证 index/coordinate 顺序。

每个数据文件保存：

```text
dataset, requested_times, returned_times
block_start_ijk, block_shape, halo, core_slices
coordinate_formula, component_order, gradient_column_map
spatial_method, temporal_method
giverny_version, metadata_hash, code_version
```

## 8. 局部滤波

### 8.1 Task 0 采用的滤波器

完整域 sharp spectral filter 无法从局部 point query 严格恢复。本 Task 改用显式的、可复现的三维离散 Gaussian convolution：

\[
\bar f(\boldsymbol x)
=\sum_{\boldsymbol n\in[-h,h]^3}
w_{\boldsymbol n}f(\boldsymbol x+\boldsymbol n\Delta),
\]

\[
w_{\boldsymbol n}
=\frac{
\exp[-|\boldsymbol n|^2/(2\sigma_g^2)]
}{
\sum_{\boldsymbol m\in[-h,h]^3}
\exp[-|\boldsymbol m|^2/(2\sigma_g^2)]
}.
\]

基线参数：

```yaml
filter:
  kind: local_discrete_gaussian
  sigma_grid: 1.0
  support_radius: 4
  boundary_output: trim_to_core
```

这里 `sigma_grid` 的单位是网格间距。核必须显式归一化，使常数场保持不变。

### 8.2 边界处理

- convolution 在完整 \(16^3\) block 上计算；
- 最终只保留距离任一 block face 至少 4 点的 \(8^3\) core；
- 不输出 halo 区域的滤波结果；
- 不使用 `mode="wrap"`，因为局部 block 不是周期域；
- 不使用 `reflect` 或 `nearest` 结果进入统计。

### 8.3 导数与滤波交换

对于空间均匀、位置无关的 convolution kernel，连续理论上

\[
\partial_j\bar u_i=\overline{\partial_j u_i}.
\]

因此本 Task 不再对 \(\bar u\) 重新做本地差分，而是对数据库返回的 gradient 使用同一个 Gaussian kernel：

\[
\overline{G}_{ij}
=\overline{\partial_j u_i}.
\]

然后定义

\[
\partial_j\bar u_i\approx\overline{G}_{ij}.
\]

这是“先由数据库求局部差分，再滤波”的离散近似，必须在 metadata 中记录为 `filter_of_fd8_gradient`。

## 9. 计算流程

### 9.1 原始 acceleration

数据库返回

\[
G_{ij}^{(8)}=(\partial_j u_i)_{\rm fd8}.
\]

逐点计算

\[
a_i^{(8)}=\sum_{j=1}^3u_jG_{ij}^{(8)}.
\]

### 9.2 滤波量

对 velocity、gradient 和原始 acceleration 使用相同 Gaussian kernel：

\[
\bar u_i=\mathcal G[u_i],
\]

\[
\overline{G}_{ij}=\mathcal G[G_{ij}^{(8)}],
\]

\[
\bar a_i=\mathcal G[a_i^{(8)}].
\]

所有结果裁剪到 \(8^3\) core。

### 9.3 Resolved acceleration

\[
\overline{\bar a}_i
=\sum_{j=1}^3\bar u_j\overline{G}_{ij}.
\]

这里 \(\overline{G}_{ij}\) 被解释为 \(\partial_j\bar u_i\) 的近似。

### 9.4 两个 work scalar

\[
W_{\rm full}
=\sum_i\bar u_i\bar a_i,
\]

\[
W_{\rm res}
=\sum_i\bar u_i\overline{\bar a}_i.
\]

## 10. Regime 与不确定性

### 10.1 基本分类

| 输出码 | 类别 | \(W_{\rm full}\) | \(W_{\rm res}\) |
|---:|---|---:|---:|
| 0 | boundary/uncertain | 任一量接近 0 或方法不稳定 | 任一量接近 0 或方法不稳定 |
| 1 | Q1 | 正 | 正 |
| 2 | Q2 | 正 | 负 |
| 3 | Q3 | 负 | 正 |
| 4 | Q4 | 负 | 负 |

### 10.2 数值 boundary

分别定义

\[
\epsilon_{\rm full}
=\max(\epsilon_{abs},
      \epsilon_{rel}\,\mathrm{RMS}(W_{\rm full})),
\]

\[
\epsilon_{\rm res}
=\max(\epsilon_{abs},
      \epsilon_{rel}\,\mathrm{RMS}(W_{\rm res})).
\]

任一 work 位于对应的 \([-\epsilon,+\epsilon]\) 内时输出 0，不强行分类。

### 10.3 FD6/FD8 方法稳定性

额外查询 `fd6noint` gradient，用于同一完整 100-snapshot time series。重复计算全部派生量，得到

```text
regime_fd8
regime_fd6
```

规则：

- 两种方法给出相同 Q1–Q4 且都不在数值 boundary：保留该 regime；
- 任一方法为 boundary，或 FD6/FD8 给出不同象限：最终标记 0；
- 报告 `fd6_fd8_disagreement_fraction`；
- 保存 \(W^{fd8}-W^{fd6}\) 的 RMS、分位数和相对误差。

这一检查不能消除相对谱导数约 7% RMS 的系统误差，但能识别最容易发生符号翻转的点。

## 11. 无散诊断

数据库 gradient 不保证逐点 divergence 为零。计算

\[
D=G_{xx}+G_{yy}+G_{zz}
=\partial_xu_x+\partial_yu_y+\partial_zu_z.
\]

报告：

- divergence mean、RMS、maximum；
- \(D\) 相对对角 gradient RMS 的归一化值；
- FD6 与 FD8 divergence 的差异；
- Gaussian-filtered divergence；
- divergence 与 regime disagreement 的相关性。

本 Task 不对数据库 gradient 做投影修正，因为单独投影 velocity 而不一致地修正 gradient 会破坏两者的配套关系。若 divergence 超出合理范围，应检查列映射和坐标，而不是静默修改数据。

## 12. 项目结构

```text
JHU_DATA/
├── pyproject.toml
├── configs/
│   └── task0.yaml
├── src/jhtdb_regimes/
│   ├── config.py
│   ├── cli.py
│   ├── jhtdb_client.py
│   ├── grid.py
│   ├── storage.py
│   ├── filtering.py
│   ├── acceleration.py
│   ├── regimes.py
│   └── quality.py
├── tests/
│   ├── test_grid_mapping.py
│   ├── test_gradient_columns.py
│   ├── test_filtering.py
│   ├── test_acceleration.py
│   ├── test_regimes.py
│   ├── test_jhtdb_mock.py
│   └── test_jhtdb_online.py
└── data/
    ├── raw/
    ├── derived/
    └── reports/
```

本版本删除完整域 FFT、Leray projection、distributed FFT 和 nonlinear dealiasing 模块；这些留给获得正式 token 和更大计算资源后的后续 Task。

## 13. 输出

### 13.1 Raw data

```text
velocity(time, component, z, y, x)
gradient_fd8(time, velocity_component, derivative_axis, z, y, x)
gradient_fd6(time, velocity_component, derivative_axis, z, y, x)  # audit
```

### 13.2 Derived core data

```text
velocity_bar(time, component, z_core, y_core, x_core)
gradient_bar_fd8(time, velocity_component, derivative_axis, z_core, y_core, x_core)
a_bar(time, component, z_core, y_core, x_core)
a_barbar(time, component, z_core, y_core, x_core)
work_full(time, z_core, y_core, x_core)
work_resolved(time, z_core, y_core, x_core)
regime_fd8(time, z_core, y_core, x_core)
regime_fd6(time, z_core, y_core, x_core)
regime_robust(time, z_core, y_core, x_core)
divergence_fd8(time, z_core, y_core, x_core)
```

### 13.3 报告

每次运行生成 JSON 和 Markdown 报告，包含：

- 请求点数、批次数、串行请求时间；
- 请求时间与返回时间；
- 坐标和 gradient column mapping；
- filter kernel 的完整权重或 hash；
- FD6/FD8 work 差异；
- divergence 统计；
- 每个 snapshot 的 Q1–Q4、boundary 和 disagreement fractions；
- snapshot-level bootstrap confidence interval；
- 软件和 metadata 版本。

## 14. 测试

### 14.1 Grid mapping

- index 到坐标的映射正确；
- point flatten/unflatten 后恢复原 `(z,y,x)`；
- x/y/z 和 ux/uy/uz 不置换；
- 4000 + 96 两个 batch 合并后正好得到 \(16^3\) 点，无重复或缺失。

### 14.2 Gradient columns

用人工 DataFrame 列名验证 \(G_{ij}=\partial_j u_i\) 映射。缺列、重复列或未知列必须失败。

### 14.3 Filter

- 常数场滤波后不变；
- kernel 权重和为 1；
- impulse response 与核一致；
- halo 不足时失败；
- 输出严格为中心 \(8^3\) core；
- 测试能发现错误使用 `wrap`/`reflect` 产生的边缘污染。

### 14.4 Acceleration

用线性/解析速度和梯度 fixture 验证指标缩并：

\[
a_i=\sum_j u_jG_{ij}.
\]

测试必须能发现把 \(G_{ij}\) 误作 \(G_{ji}\) 的 bug。

### 14.5 Regime

- Q1–Q4 的四种符号组合；
- 数值 boundary；
- FD6/FD8 一致时保留；
- 符号不一致时标记 uncertain；
- NaN/Inf 永远不进入 Q1–Q4。

### 14.6 API mock

- testing token 强制 batch <=4000；
- 请求严格串行；
- retry 也不能产生并发；
- 返回时间、point count 或列数错误时失败；
- token 不出现在日志。

### 14.7 在线 smoke test

只查询少量网格点和 1–2 个时间：

- `field/none` 返回三分量 velocity；
- `gradient/fd8noint` 返回 9 个具名 gradient；
- 时间与坐标正确；
- smoke test 通过后才允许提交完整 Task 0 time-series 请求。

## 15. 命令行

```text
jhtdb-regimes plan      configs/task0.yaml
jhtdb-regimes smoke     configs/task0.yaml
jhtdb-regimes fetch     configs/task0.yaml
jhtdb-regimes validate  configs/task0.yaml --stage raw
jhtdb-regimes compute   configs/task0.yaml
jhtdb-regimes classify  configs/task0.yaml
jhtdb-regimes report    configs/task0.yaml
jhtdb-regimes run       configs/task0.yaml
```

`plan` 必须在任何网络请求前显示：点数、时间数、预计请求数、每批 point×time、预计响应大小以及 testing-token 串行策略。

## 16. 验收标准

Task 0 完成需要：

1. 使用 testing token，所有请求点数不超过 4000；
2. 所有网络请求严格串行，并等待上一个完成；
3. 成功返回 100 个目标真实存储时间；若服务缺失个别时间，manifest 必须明确列出，不得替换或静默丢弃；
4. velocity 和 `fd8noint` gradient 在完整 \(16^3\) block 上无缺失；
5. gradient 列名到 \(G_{ij}\) 的映射经过测试；
6. Gaussian kernel、halo 和 core 均符合配置；
7. 成功计算 \(\bar u\)、\(\bar a\)、\(\overline{\bar a}\)、两个 work 和 regimes；
8. 输出 FD6/FD8 regime disagreement；
9. 对 near-zero 或方法不稳定点使用 boundary/uncertain，不强行分类；
10. 输出数据库 gradient 的 divergence 诊断；
11. Q1–Q4 加 uncertain 的比例为 1；
12. 所有离线测试和小规模在线 smoke test 通过；
13. 报告明确写明“database local finite-difference gradient；约 7% RMS gradient error；local Gaussian filter；非全域 spectral result”。

## 17. 后续升级路径

获得个人 token 和更大内存/HPC 后，再建立独立 Task：

1. 下载完整 \(1024^3\) 周期快照；
2. 使用全域 spectral derivative 和 Leray projection；
3. 实现正式 Gaussian-smoothed sharp spectral filter；
4. 比较本 Task 的 `fd8noint` regimes 与 spectral regimes；
5. 定量得到 database-gradient approximation 对 \(\bar a\)、\(\overline{\bar a}\) 和 Q1–Q4 的实际影响。

## 18. 参考资料

- [JHTDB Database Access 与 testing token](https://turbulence.idies.jhu.edu/database)
- [JHTDB Spatial and Temporal Methods](https://turbulence.idies.jhu.edu/database/query/methods)
- [JHTDB isotropic1024 README](https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic1024.pdf)
- [JHTDB 公告：testing token 请求必须串行](https://turbulence.idies.jhu.edu/announcements)
- [官方 Giverny](https://github.com/sciserver/giverny)

## 19. 已部署脚本与运行方法

项目已经按上述设计实现于：

```text
task0.py
task0.bat
configs/task0.yaml
src/jhtdb_regimes/
tests/
run_task0.ps1
```

当前安装的是官方 `givernylocal 3.6.2`。若环境变量 `JHTDB_TOKEN` 未设置，客户端会从官方 metadata 读取内置 testing token，只在运行内存中使用，不写入配置、缓存、manifest 或日志。

已验证在线返回列：

```text
velocity: ux, uy, uz
gradient: duxdx, duxdy, duxdz,
          duydx, duydy, duydz,
          duzdx, duzdy, duzdz
```

运行：

```powershell
python task0.py plan
python task0.py smoke
python task0.py fetch
python task0.py validate
python task0.py compute
```

也可以使用 PowerShell 启动器：

```powershell
./run_task0.ps1 plan
./run_task0.ps1 smoke
./run_task0.ps1 fetch
./run_task0.ps1 validate
./run_task0.ps1 compute
```

或者执行完整串行流水线：

```powershell
./run_task0.ps1 run
```

正式下载前建议保留 `data/cache/`。它包含逐请求断点，重新运行 `fetch` 时会校验并复用成功结果。
