# Task 0 独立验证报告

总体结果：**PASS**

## 检查项

- PASS — `raw_sha256`
- PASS — `times_exact`
- PASS — `indices_exact`
- PASS — `all_raw_finite`
- PASS — `all_derived_finite`
- PASS — `fd8_matches_velocity_stencil`
- PASS — `fd6_matches_velocity_stencil`
- PASS — `kernel_normalized_symmetric`
- PASS — `direct_3d_filter_matches`
- PASS — `derived_algebra`
- PASS — `primary_regime_exact`
- PASS — `audit_regime_exact`
- PASS — `robust_regime_exact`
- PASS — `regime_range`

## 关键误差

- database FD8 vs local FD8 relative RMS: `1.549752e-07`
- database FD6 vs local FD6 relative RMS: `1.502997e-07`
- database FD6 vs FD8 relative RMS: `1.355778e-02`
- direct 3-D filter maximum relative RMS: `2.860088e-08`
- divergence RMS / gradient RMS: `3.336055e-02`
- FD6/FD8 regime disagreement: `2.128906e-03`
- robust uncertain fraction: `4.687500e-03`

## 解释边界

本报告验证了下载完整性、网格轴顺序、数据库有限差分 stencil、局部 Gaussian 实现、派生代数和 regime 分类。它不能把数据库有限差分证明成 DNS 的全局谱导数，也不能用当前 16³ block 验证完整的 filter–derivative commutation。
