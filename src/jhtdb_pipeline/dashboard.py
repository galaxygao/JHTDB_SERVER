from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.store import open_complete_result


REGIME_LABELS = ("uncertain", "Q1: +/+", "Q2: +/-", "Q3: -/+", "Q4: -/-")
REGIME_COLORS = [
    [0.00, "#9e9e9e"], [0.199999, "#9e9e9e"],
    [0.20, "#1f77b4"], [0.399999, "#1f77b4"],
    [0.40, "#ff7f0e"], [0.599999, "#ff7f0e"],
    [0.60, "#2ca02c"], [0.799999, "#2ca02c"],
    [0.80, "#d62728"], [1.00, "#d62728"],
]
GLOBAL_TOTAL_ORDER = ("s_bar", "pi", "work_resolved", "work_full")
GLOBAL_TOTAL_LABELS = ("ΣS̄", "ΣΠ", "ΣW_res", "ΣW_full")
SBAR_METRIC_SPECS = (
    ("identity_residual_rms", "能量等式残差 RMS"),
    ("s_bar_rel_self", "|ΣS̄| / Σ|S̄|"),
    ("s_bar_vs_pi_net", "|ΣS̄| / |ΣΠ|"),
)


def complete_result_paths(result_root: Path) -> list[Path]:
    if not result_root.is_dir():
        return []
    return sorted(
        path
        for path in result_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "COMPLETE").is_file()
    )


def extract_slice(array, component: int, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return np.asarray(array[component, :, :, index])
    if axis == "y":
        return np.asarray(array[component, :, index, :])
    if axis == "z":
        return np.asarray(array[component, index, :, :])
    raise ValueError(f"unknown axis {axis}")


def extract_scalar_slice(array, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return np.asarray(array[:, :, index])
    if axis == "y":
        return np.asarray(array[:, index, :])
    if axis == "z":
        return np.asarray(array[index, :, :])
    raise ValueError(f"unknown axis {axis}")


def extract_gradient_slice(
    array, velocity_component: int, derivative_component: int, axis: str, index: int
) -> np.ndarray:
    if axis == "x":
        return np.asarray(array[velocity_component, derivative_component, :, :, index])
    if axis == "y":
        return np.asarray(array[velocity_component, derivative_component, :, index, :])
    if axis == "z":
        return np.asarray(array[velocity_component, derivative_component, index, :, :])
    raise ValueError(f"unknown axis {axis}")


def spatial_axis_length(array, axis: str) -> int:
    return int(array.shape[{"z": -3, "y": -2, "x": -1}[axis]])


def full_index_to_crop(index: int, axis: str, crop_start, crop_shape) -> int | None:
    component = {"x": 0, "y": 1, "z": 2}[axis]
    local = index - int(crop_start[component])
    return local if 0 <= local < int(crop_shape[component]) else None


def _global_totals_figure(report: dict):
    totals = report["global_totals"]
    return go.Figure(
        go.Bar(
            x=list(GLOBAL_TOTAL_LABELS),
            y=[totals[name] for name in GLOBAL_TOTAL_ORDER],
            customdata=[totals[name] for name in GLOBAL_TOTAL_ORDER],
            hovertemplate="%{x}: %{customdata:.8e}<extra></extra>",
        )
    ).update_layout(title=f"Global net totals ({report['scope']})")


def _scientific_text(value) -> str:
    if value is None:
        return "不可定义"
    return f"{float(value):.6e}"


def sbar_metric_rows(report: dict) -> list[dict[str, str]]:
    metrics = report.get("metrics", {})
    rows = []
    for key, label in SBAR_METRIC_SPECS:
        metric = metrics.get(key, {})
        detail = metric.get("error") or ""
        if key == "identity_residual_rms" and metric.get("maximum_abs") is not None:
            detail = f"maximum_abs={_scientific_text(metric['maximum_abs'])}"
        rows.append(
            {
                "判据": label,
                "值": _scientific_text(metric.get("value")),
                "阈值（≤）": _scientific_text(metric.get("threshold")),
                "状态": "通过" if metric.get("passed") else "失败",
                "说明": detail,
            }
        )
    return rows


def energy_identity_residual(
    work_full: np.ndarray,
    work_resolved: np.ndarray,
    pi: np.ndarray,
    s_bar: np.ndarray,
) -> np.ndarray:
    return work_full - work_resolved + pi - s_bar


def _symmetric_color_limit(values: np.ndarray, percentile: float = 100.0) -> float:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    finite_magnitudes = np.abs(values[np.isfinite(values)])
    if not finite_magnitudes.size:
        return 1.0
    limit = float(np.percentile(finite_magnitudes, percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(finite_magnitudes))
    return limit if limit > 0.0 else 1.0


def _symlog_transform(values: np.ndarray, linear_threshold: float) -> np.ndarray:
    if linear_threshold <= 0.0:
        raise ValueError("linear_threshold must be positive")
    values = np.asarray(values)
    return np.sign(values) * np.log1p(np.abs(values) / linear_threshold)


def _continuous_figure(
    values: np.ndarray,
    title: str,
    *,
    signed: bool = True,
    color_percentile: float = 100.0,
    scale_mode: str = "linear",
    color_limit: float | None = None,
):
    if scale_mode not in {"linear", "symlog"}:
        raise ValueError("scale_mode must be 'linear' or 'symlog'")
    if not signed and scale_mode != "linear":
        raise ValueError("symlog color scaling requires signed=True")
    stride = max(1, values.shape[0] // 512)
    shown = values[::stride, ::stride]
    displayed = shown
    kwargs = {}
    if signed:
        limit = (
            _symmetric_color_limit(shown, color_percentile)
            if color_limit is None
            else float(color_limit)
        )
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("color_limit must be finite and positive")
        displayed_limit = limit
        if scale_mode == "symlog":
            linear_threshold = limit * 0.01
            displayed = _symlog_transform(shown, linear_threshold)
            displayed_limit = float(
                _symlog_transform(np.asarray(limit), linear_threshold)
            )
        kwargs = {"zmin": -displayed_limit, "zmax": displayed_limit}
    figure = px.imshow(
        displayed,
        origin="lower",
        color_continuous_scale="RdBu_r" if signed else "Viridis",
        aspect="equal",
        **kwargs,
    )
    if signed and scale_mode == "symlog":
        raw_ticks = limit * np.asarray([-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0])
        transformed_ticks = _symlog_transform(raw_ticks, linear_threshold)
        figure.update_traces(
            customdata=shown,
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{customdata:.6g}<extra></extra>",
        )
        figure.update_coloraxes(
            colorbar={
                "tickvals": transformed_ticks.tolist(),
                "ticktext": [f"{value:.3g}" for value in raw_ticks],
                "title": "value (SymLog)",
            }
        )
    figure.update_layout(title=title)
    return figure


def _regime_figure(values: np.ndarray, title: str):
    shown = np.asarray(values, dtype=np.uint8)
    labels = np.asarray(REGIME_LABELS, dtype=object)[np.clip(shown, 0, 4)]
    figure = go.Figure(
        go.Heatmap(
            z=shown,
            customdata=labels,
            colorscale=REGIME_COLORS,
            zmin=-0.5,
            zmax=4.5,
            colorbar={"tickvals": [0, 1, 2, 3, 4], "ticktext": list(REGIME_LABELS)},
            hovertemplate="x=%{x}<br>y=%{y}<br>code=%{z}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(title=title)
    figure.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    return figure


def main() -> None:
    st.set_page_config(page_title="JHTDB SciServer viewer", layout="wide")
    cfg = load_config(os.environ.get("JHTDB_PIPELINE_CONFIG", "configs/pipeline.yaml"))
    st.title("JHTDB 中心周期域结果（服务器只读）")
    paths = complete_result_paths(cfg.result_root)
    if not paths:
        st.info("persistent 中还没有带 COMPLETE 标记的正式结果。")
        return
    selected = st.sidebar.selectbox("Result", paths, format_func=lambda path: path.name)
    result = open_complete_result(selected)
    st.caption(
        f"{selected} | frame={result.attrs['time_index']} | "
        f"sigma={result.attrs['sigma_grid']}"
    )
    page = st.sidebar.radio(
        "页面",
        ("速度对比", "梯度对比", "Work 与 regime", "Π 与 S̄", "全域 S̄ QA"),
    )
    axis = st.sidebar.selectbox("切片法向", ("z", "y", "x"))
    index = None
    if page != "全域 S̄ QA":
        indexed_field = (
            result["work_full"]
            if page in ("Work 与 regime", "Π 与 S̄")
            else result["velocity"]
        )
        length = spatial_axis_length(indexed_field, axis)
        scope = "全域" if page in ("Work 与 regime", "Π 与 S̄") else "中心域"
        index = st.sidebar.slider(
            f"{scope}切片 index", 0, length - 1, length // 2
        )

    if page == "速度对比":
        component = st.sidebar.selectbox(
            "速度分量", (0, 1, 2), format_func=lambda value: ("ux", "uy", "uz")[value]
        )
        raw = extract_slice(result["velocity"], component, axis, index)
        filtered = extract_slice(result["velocity_bar"], component, axis, index)
        limit = _symmetric_color_limit(np.stack((raw, filtered)))
        left, right = st.columns(2)
        left.plotly_chart(
            _continuous_figure(raw, "velocity", color_limit=limit),
            use_container_width=True,
        )
        right.plotly_chart(
            _continuous_figure(filtered, "velocity_bar", color_limit=limit),
            use_container_width=True,
        )
    elif page == "梯度对比":
        component = st.sidebar.selectbox(
            "速度分量 i", (0, 1, 2), format_func=lambda value: ("ux", "uy", "uz")[value]
        )
        derivative = st.sidebar.selectbox(
            "求导方向 j", (0, 1, 2), format_func=lambda value: ("x", "y", "z")[value]
        )
        label = st.sidebar.radio("梯度色标", ("线性", "SymLog"), horizontal=True)
        percentile = st.sidebar.slider("色标覆盖分位数 (%)", 90.0, 100.0, 99.0, 0.5)
        mode = "linear" if label == "线性" else "symlog"
        raw = extract_gradient_slice(result["gradient"], component, derivative, axis, index)
        filtered = extract_gradient_slice(result["gradient_bar"], component, derivative, axis, index)
        limit = _symmetric_color_limit(np.stack((raw, filtered)), percentile)
        left, right = st.columns(2)
        left.plotly_chart(
            _continuous_figure(
                raw,
                "gradient",
                color_percentile=percentile,
                scale_mode=mode,
                color_limit=limit,
            ),
            use_container_width=True,
        )
        right.plotly_chart(
            _continuous_figure(
                filtered,
                "gradient_bar",
                color_percentile=percentile,
                scale_mode=mode,
                color_limit=limit,
            ),
            use_container_width=True,
        )
    elif page == "Work 与 regime":
        full = extract_scalar_slice(result["work_full"], axis, index)
        resolved = extract_scalar_slice(result["work_resolved"], axis, index)
        left, right = st.columns(2)
        left.plotly_chart(_continuous_figure(full, "work_full"), use_container_width=True)
        right.plotly_chart(_continuous_figure(resolved, "work_resolved"), use_container_width=True)
        regime_index = full_index_to_crop(
            index, axis, cfg.crop_start, cfg.crop_shape
        )
        if regime_index is None:
            st.info("当前全域切片位于中心 crop 之外，没有持久化 regime。")
        else:
            codes = extract_scalar_slice(result["regime"], axis, regime_index)
            st.plotly_chart(_regime_figure(codes, "regime"), use_container_width=True)
        st.json(dict(result.attrs.get("occupancy", {})))
    elif page == "Π 与 S̄":
        if "pi" not in result or "s_bar" not in result:
            st.warning("该旧版结果不包含 pi/s_bar；需要用新版物理流水线重新计算。")
        else:
            pi = extract_scalar_slice(result["pi"], axis, index)
            s_bar = extract_scalar_slice(result["s_bar"], axis, index)
            limit = _symmetric_color_limit(np.stack((pi, s_bar)), 99.0)
            left, right = st.columns(2)
            left.plotly_chart(
                _continuous_figure(pi, "Π = τᵢⱼ ∂ⱼūᵢ", color_limit=limit),
                use_container_width=True,
            )
            right.plotly_chart(
                _continuous_figure(
                    s_bar, "S̄ = ∂ⱼ(ūᵢτᵢⱼ)", color_limit=limit
                ),
                use_container_width=True,
            )
            st.caption(
                "式 (2) 符号约定：W_full = W_resolved − Π + S̄；"
                "常见 LES 定义 Π_conventional = −τ:S = −Π。"
            )
            full = extract_scalar_slice(result["work_full"], axis, index)
            resolved = extract_scalar_slice(result["work_resolved"], axis, index)
            residual = energy_identity_residual(full, resolved, pi, s_bar)
            st.plotly_chart(
                _continuous_figure(
                    residual,
                    "当前切片能量等式残差 W_full − W_resolved + Π − S̄",
                    color_percentile=99.0,
                ),
                use_container_width=True,
            )
            st.json(dict(result.attrs.get("decomposition", {})))
    else:
        st.header("全域 S̄ QA")
        s_bar_qa = selected / "s_bar_qa.json"
        if s_bar_qa.is_file():
            report = json.loads(s_bar_qa.read_text(encoding="utf-8"))
            if report.get("passed"):
                st.success("全域 S̄ QA：通过")
            else:
                st.error("全域 S̄ QA：失败；正式数据仍保留用于诊断")
            st.caption(
                f"{report.get('identity', 'work_full = work_resolved - pi + s_bar')} | "
                f"scope={report.get('scope')} | points={report.get('point_count', 0):,}"
            )
            columns = st.columns(3)
            for column, (key, label) in zip(columns, SBAR_METRIC_SPECS):
                metric = report.get("metrics", {}).get(key, {})
                column.metric(label, _scientific_text(metric.get("value")))
                status = "通过" if metric.get("passed") else "失败"
                column.caption(
                    f"阈值 ≤ {_scientific_text(metric.get('threshold'))} · {status}"
                )
            st.dataframe(
                sbar_metric_rows(report), hide_index=True, use_container_width=True
            )
            st.plotly_chart(
                _global_totals_figure(report), use_container_width=True
            )
            with st.expander("查看 s_bar_qa.json 原始报告"):
                st.json(report)
        else:
            st.warning(
                "当前正式结果没有 s_bar_qa.json。请选择已完成的 schema v4 结果，"
                "或对完整 v4 结果运行 qa-sbar。正在计算的 staging 不会进入 GUI。"
            )
        st.subheader("完整性与其他 QA 记录")
        for filename in ("manifest.json", "qa.json", "divergence.json", "COMPLETE"):
            path = selected / filename
            st.subheader(filename)
            if path.is_file():
                try:
                    st.json(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    st.code(path.read_text(encoding="utf-8"))
            else:
                st.warning("missing")

    st.caption("本 GUI 只读取服务器 persistent 正式结果，不写数据、不访问本地文件。")


if __name__ == "__main__":
    main()
