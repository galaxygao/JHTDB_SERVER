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
    page = st.sidebar.radio("页面", ("速度对比", "梯度对比", "Work 与 regime", "QA"))
    axis = st.sidebar.selectbox("切片法向", ("z", "y", "x"))
    index = st.sidebar.slider("中心域切片 index", 0, result["work_full"].shape[0] - 1, result["work_full"].shape[0] // 2)

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
        codes = extract_scalar_slice(result["regime"], axis, index)
        left, right = st.columns(2)
        left.plotly_chart(_continuous_figure(full, "work_full"), use_container_width=True)
        right.plotly_chart(_continuous_figure(resolved, "work_resolved"), use_container_width=True)
        st.plotly_chart(_regime_figure(codes, "regime"), use_container_width=True)
        st.json(dict(result.attrs.get("occupancy", {})))
    else:
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
