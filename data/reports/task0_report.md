# Task 0 计算报告

- 数据集：`isotropic1024coarse`
- snapshots：100
- block/core：`(16, 16, 16)` / `(8, 8, 8)`
- 梯度：`fd8noint`；审计：`fd6noint`
- FD6/FD8 disagreement：0.00212890625

## Robust regime occupancy

- uncertain: 0.00468750
- Q1: 0.56005859
- Q2: 0.00673828
- Q3: 0.00648437
- Q4: 0.42203125

## 重要限制

数据库局部有限差分梯度相对 DNS 全局谱梯度约有 7% RMS 误差；本结果使用局部 Gaussian 滤波，不是全域 spectral result。
