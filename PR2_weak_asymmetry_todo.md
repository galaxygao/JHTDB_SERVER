# PR2 — weak asymmetry 待办

> 实施状态（2026-08-31）：2a 全域正负 Π 分拆已完成，并补充 `ratio_p99=|mean(Pi)|/percentile(|Pi|,99)` 与 `ratio_max=|mean(Pi)|/max(|Pi|)`。输出 `weak_asymmetry.json/html`，提供 `compute-weak-asymmetry` 和只读 GUI 页面，并在常规 Cq 扫描中复用同一批 chunk 同步生成。已有 v1 报告升级时只读取 persistent `pi` 一次，不重算 Cq、FFT 或 JHTDB。下述“2a-扩展”按象限细分正负 Π 仍是下一项任务，尚未实现。

## 目标
在现有单帧 1024³ 数据上定量刻画 weak asymmetry：
强前向 / 逆向 patch 近乎抵消、只剩微弱净通量。全域可做，数据已在 persistent，只需加统计输出。

## 符号约定（钉死）
- Π ≡ τ_ij S_ij。**Π < 0 = 前向级串（大→小）；Π > 0 = 逆向 / backscatter。**
- 数据 ΣΠ = −64M < 0 → 3D 净前向，自洽。写论文/画图必须标这句。

---

## 2a：正负 Π 分拆（回答"弱不弱"）

补以下全域标量：

```python
Pi_pos_sum  = Pi[Pi > 0].sum()          # backscatter（逆向）总量
Pi_neg_sum  = Pi[Pi < 0].sum()          # forward（前向）总量
Pi_pos_frac = (Pi > 0).mean()           # 逆向 patch 占比
Pi_neg_frac = (Pi < 0).mean()           # 前向 patch 占比
Pi_mean     = Pi.mean()                 # 净通量 ⟨Π⟩
Pi_rms      = np.sqrt((Pi**2).mean())   # 涨落 RMS
asymmetry_index = Pi_mean / Pi_rms      # 核心指标：净 vs RMS
ratio_p99 = abs(Pi_mean) / percentile(abs(Pi), 99)
ratio_max = abs(Pi_mean) / max(abs(Pi))
```

自检：`Pi_pos_sum + Pi_neg_sum == Pi.sum()`（=ΣΠ=−64M）。

判读：
- Pi_pos_sum、Pi_neg_sum 量级接近、相减剩一小截 → weak asymmetry 本来面目。
- `asymmetry_index = ⟨Π⟩/√⟨Π²⟩` 是唯一能和 proposal 2D（≈0.025，即 1/40）同口径对比的数，越小越"弱"。
- `ratio_p99` 和 `ratio_max` 分别比较净通量与强尾部/极值幅度；两者使用 `|mean(Pi)|`，不携带净级串方向符号。
- 不要用 |ΣΠ|/Σ|Π| 对标 proposal，那是错口径。

---

## 2a-扩展：按象限做正负分拆（接现有 C_q）

对 4 象限分别统计正负 Π：

```python
for q in [Q1, Q2, Q3, Q4]:
    Pi_pos_sum | q, Pi_neg_sum | q, Pi_rms | q
```

目的：把 weak asymmetry（正负抵消）和象限贡献 C_q 连起来，
看某象限 C_q 小到底是"本来弱"还是"强正强负抵消掉了"。现有 C_q 的自然延伸。

---

## 执行顺序
1. 2a 正负分拆 —— 最先做
2. 2a 按象限细化 —— 接现有 C_q

全部单帧、数据在盘、无需重新取数。
