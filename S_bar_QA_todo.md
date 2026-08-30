# S̄ 全域 QA — 待办与理由

> 后续变更：schema v5 已把 `regime` 从中心 `512³` 扩展为全周期域 `1024³`。下文关于 schema v4 保持中心 regime 的内容是当时的实施边界；已有 v4 可直接复用 persistent 全域 `work_full/work_resolved` 快速补齐，新计算与 `single-frame` 均永久保存全域 regime。S̄ 的 `|ΣS̄|/Σ|S̄|` 绝对归一化 QA 后续已删除，只保留相对联合 RMS 能量等式和相对净 `Π` 判据。Cq 已改为零阈值符号 Q1–Q4 完备划分，并执行 `C1+...+C4=mean(pi)` closure。

## 目标
在**全周期域**上验证 S̄ = ∂ⱼ(ūᵢτᵢⱼ) 的散度归零性质，确认其残余不污染净级串 ⟨Π⟩，并把四个量的全域总量可视化。

## 要做的事

### 1. 主 pipeline：四个场全域存 persistent
W_full、W_res、Π、S̄ 都保留**全域**（不再 crop 到中心子域），存到 **persistent**（不用 scratch）。

理由：
- 这四个是要长期复用、进 QA 和 C_q 分解、还驱动可视化的正式结果，属于"最终结果"，本就该放有备份的 persistent。
- 全域存下来后，净量判据（ΣS̄、ΣΠ、vs_Pi_net）**随时可事后补算**，不再依赖跑的时候有没有算好标量、也不用为漏存而重跑。
- C_q 分解可直接在全域做，消除子域边界污染与代表性问题。

### 2. 独立 QA 脚本，读全域四场，算两项判据 + 出一张柱状图
和主 pipeline 解耦：主 pipeline 只负责"全域存四场"；QA 单独跑、可重复跑、可对已有结果补跑。

## 两项判据

| 指标 | 表达式 | 回答 | 通过阈值 |
|---|---|---|---|
| 逐点预算残差 | `W_full = W_res − Π + S̄` 的 residual RMS / 四场联合 RMS | pipeline 逐点是否正确（**数值**） | ~1e-4 以下 |
| `S_bar_vs_Pi_net` | \|ΣS̄\| / \|ΣΠ\| | S̄ 残余会不会污染净级串（**科学**）★决定性 | << 1（~1e-2 以下） |

判读关键：净 ⟨Π⟩ 本就很小（weak asymmetry），若 ΣS̄ 残余与 ΣΠ 同量级，C_q 分解分辨出的"净贡献"可能混入 S̄ 漏项，因此直接使用净量之比，不再使用容易稀释残差的 Σ\|S̄\| 绝对归一化判据。
- 分母用**同一口径**（都不带绝对值的全域求和）比，避免用 Σ\|Π\| 稀释掉 B。

## 柱状图
四根柱：S̄、Π、W_res、W_full 各自的全域**净量**（ΣS̄、ΣΠ、ΣW_res、ΣW_full，不带绝对值）。

- 口径取**净量 Σ·**，与 `S_bar_vs_Pi_net` 判据同口径。
- 逐点活跃程度已有场图，柱状图不重复；只做总量。
- 预期一眼可读：ΣS̄≈0、ΣΠ 是很小的净级串、ΣW_full≈ΣW_res（差值 = −ΣΠ+ΣS̄）。

## 不受影响、不用改的部分
- C_q 分解与 `ΣC_q = ⟨Π⟩` 自检：Π 非散度，全域直接算即可。

## 实施方案（2026-08-29，代码修改前）

### A. 结果 schema 与空间口径

1. schema 升级到 v4。`velocity`、`gradient`、`velocity_bar`、`gradient_bar` 和 `regime` 继续保存中心 `512³`；`work_full`、`work_resolved`、`pi`、`s_bar` 改为完整周期域 `1024³`。
2. Zarr attrs 和 manifest 明确记录两类字段的 `origin_xyz`、`shape_xyz` 与 `scope`，四个全域字段标记为 `full_domain`，避免 dashboard 或后处理继续把它们误认为中心坐标。
3. 正式结果仍先写 `persistent/results/.staging`，所有字段和 QA 完成后才原子替换旧结果并创建 `COMPLETE`；不会在原 v3 目录里直接扩大数组，以免中断破坏已有结果。
4. `regime` 暂时维持中心域，符合本待办“不改 C_q 分解”的范围；后续若要做全域逐点 C_q/regime，再单独扩 schema，不能在本次顺带扩大 persistent 占用。

### B. 主 pipeline 如何生成四个全域场

1. 保持现有完整周期域 FFT、谱导数和滤波顺序不变，只把四个标量的写入目标从 `_accumulate_center/_accumulate_crop` 改成 persistent staging 中的全域 Zarr 数组。
2. 按 slab/chunk 流式累加，不在 RAM 中构造 `1024³ × 4` 副本，也不新增四个 scratch memmap；计算临时量仍复用现有 `derivative`、`acceleration`、`sgs_transport` 和两个滤波 buffer。
3. 中心 v3 旧字段存在时，重算产生的中心重叠区必须逐 chunk 与旧 `work_full/work_resolved/pi/s_bar` 比较；通过后才允许提交 v4。这既复用已有 persistent 结果，也给新全域写法增加回归证据。
4. 新鲜 v4 计算继续执行原始/滤波速度的全域无散检查；全域四场写完后不在主 pipeline 内重复实现 S̄ 总量判据，而是调用同一个独立 QA 计算入口，避免两套公式漂移。

### C. 既有结果复用优先级与补算策略

按以下顺序探测，每项都必须验证 time、sigma、shape、dtype 和 input manifest hash，不能只因文件存在就复用：

1. **已有 persistent v4 四场**：直接运行独立 QA，不做任何物理重算。
2. **未完成但带 checkpoint 的 persistent v4 staging**：按已完成阶段续跑；checkpoint 记录阶段、输入 hash、sigma 和各输出状态。只复用完整阶段，不猜测一个累加到一半的数组。
3. **temporary 的尺度 workspace**：若 `filtered_velocity.f32` 存在，尺寸正确，且其中心区与 v3 `velocity_bar` 数值一致，则复用它，跳过三分量高斯滤波；其余会被后续循环覆盖的临时文件不能当作最终量复用。
4. **temporary 的 `velocity_cache.zarr`**：验证状态与 manifest hash 后直接复用，绝不重新访问 JHTDB。这是成功运行清理 workspace 后仍最可能保留、也最重要的复用路径。
5. **已有 persistent v3 中心结果**：不能补出中心外的四场，因此不能假装完成全域 QA；仅用于中心重叠校验，并保留到 v4 staging 全部通过后再替换。
6. 若 raw cache 已过期/丢失且没有完整 v4 staging，才需要重新 fetch。命令必须先报告缺少的资产和预计工作量，不自动静默下载。

为落实上述顺序，新增独立的 `backfill-full-fields` 命令。它只处理已有正式 v3 结果：优先复用 workspace，其次复用 raw cache；默认不访问 JHTDB。缺少可复用 raw 数据时 fail closed，并提示先运行 `cache/validate-input`。常规 `process-center` 生成新结果时也使用同一套 checkpoint 和复用判断。

### D. 独立 S̄ QA 命令

新增 `qa-sbar --time-index N [--sigma-grid S]`，只读取当前 complete schema 的四个全域字段，逐 Zarr chunk、用 float64 累加，输出到同一正式结果目录：

- `s_bar_qa.json`：记录字段 hash/manifest hash、点数、四个净和与平方和/RMS、能量等式相对联合 RMS、绝对诊断残差、净量比、阈值、逐项 passed 和总 passed；
- `s_bar_global_totals.html`：Plotly 四柱图，柱为 `ΣS̄`、`ΣΠ`、`ΣW_res`、`ΣW_full`，hover 同时显示科学计数法数值；
- `qa.json`：只写入上述报告的摘要和报告 hash，保留已有 divergence、occupancy 等内容。

指标采用以下机器可判定定义：

```text
residual = work_full - (work_resolved - pi + s_bar)
identity_residual_rms = sqrt(Σ residual² / N)
joint_energy_rms = sqrt(Σ(work_full² + work_resolved² + pi² + s_bar²) / N)
identity_relative_residual_rms = identity_residual_rms / joint_energy_rms
s_bar_vs_pi_net = |Σ s_bar| / |Σ pi|
```

pass 使用 `identity_relative_residual_rms` 和 `s_bar_vs_pi_net`，默认阈值分别为 `1e-4`、`1e-2`，写入 config 的 validation 段并严格校验。绝对 residual RMS 和最大绝对值仍保留为诊断量，但不单独决定 pass。报告同时缓存各场平方和/RMS；只改阈值时可直接重新判定并更新 JSON/HTML、QA、manifest 和 attrs，无需重新扫描全域数组。若净 `pi` 分母为零：分子也为零则比值记为 `0`，否则记为 `null`、判定失败并在报告注明原因，JSON 中不写 NaN/Inf。

旧 v3 允许用同一计算器输出 `scope=center_crop` 的诊断预览，但不得命名为全域 QA、不得升级为 v4，也不得让正式 `qa-sbar` 返回成功。

### E. Dashboard 与坐标处理

1. sidebar 的切片 index 改为按当前页面字段 shape 生成：速度/梯度/regime 使用中心 `512³`，四个 work/Π/S̄ 页面使用全域 `1024³`。
2. 全域四场切片显示全域 index；与中心字段联合显示时，把全域数组裁到 `crop_start/crop_shape` 后再比较，禁止直接用同一局部 index 读取两个不同坐标域。
3. QA 页面读取 `s_bar_qa.json`，展示两项判据、scope、pass/fail 和同一张四柱图；dashboard 保持只读，不在页面加载时现场扫描 `1024³`。

### F. 容量、提交与失败恢复

1. 当前中心结果约 `14.125 GiB/尺度`（未压缩）；v4 为中心非四场约 `12.125 GiB` 加四个全域场 `16 GiB`，合计约 `28.125 GiB/尺度`，三个尺度约 `84.375 GiB`，尚未计 manifest、状态文件和安全余量。
2. 100 GB 账户配额若按十进制折合约 `93.1 GiB`，三个 v4 尺度与当前 `15 GiB` reserve 不能同时满足。`plan` 与 preflight 必须显示单尺度、批量和“保留旧 v3 直到提交”的峰值；空间不足时拒绝开算，不能靠降低 reserve 掩盖问题。实际执行前应选择较少 sigma、增加 quota，或确认压缩后的实测占用与保留策略。
3. 每完成一个可独立复用阶段就原子写 checkpoint；失败保留 staging 和 workspace，不删除旧 complete。只有 v4 提交成功后，才按 `cleanup_scratch_on_success` 清理该尺度 workspace。
4. QA 重跑只读四个全域数组并重写小型 JSON/HTML，不触碰大数组；QA 失败不删除数据，但 formal status/QA 总状态必须显示失败，避免下游把未通过结果当作已验收。

### G. 测试与验收顺序

1. 小网格端到端测试：验证四场 shape 为全域、其他字段仍为 crop，并逐点验证能量等式。
2. 复用测试：已有 v4 只跑 QA；v3 + 有效 workspace 跳过滤波；v3 + raw cache 不 fetch；缺少 raw cache 时明确失败；中心重叠不一致时拒绝提交。
3. QA 单元测试：构造零和/非零和 S̄，覆盖两个阈值、零分母和 JSON 无 NaN/Inf。
4. dashboard 测试：中心/全域 index 映射和柱图数据顺序正确。
5. 容量测试：核对 `28.125 GiB/尺度` 的公式和三尺度 preflight。
6. 完整回归通过后更新 README、系统指南与 CLI 帮助；最后在 SciServer 先对一个 sigma 执行 backfill + QA，人工核对报告，再决定是否扩到其余尺度。
