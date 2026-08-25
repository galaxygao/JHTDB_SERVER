from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import convolve

from .config import TaskConfig, load_config
from .physics import advective_acceleration, gaussian_kernel_1d, gaussian_valid
from .verify import finite_difference_core


COMPONENTS = ("x", "y", "z")
REGIME_LABELS = ("boundary/uncertain", "Q1", "Q2", "Q3", "Q4")
REGIME_COLORS = [
    [0.00, "#9e9e9e"],
    [0.199999, "#9e9e9e"],
    [0.20, "#1f77b4"],
    [0.399999, "#1f77b4"],
    [0.40, "#ff7f0e"],
    [0.599999, "#ff7f0e"],
    [0.60, "#2ca02c"],
    [0.799999, "#2ca02c"],
    [0.80, "#d62728"],
    [1.00, "#d62728"],
]


@st.cache_resource(show_spinner="读取 raw 和 derived 数据……")
def load_arrays(raw_path: str, derived_path: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(raw_path, allow_pickle=False) as saved:
        raw = {name: saved[name] for name in saved.files}
    with np.load(derived_path, allow_pickle=False) as saved:
        derived = {name: saved[name] for name in saved.files}
    return raw, derived


@st.cache_resource(show_spinner="从速度独立重建 FD6/FD8 梯度……")
def reconstruct_gradients(raw_path: str, spacing: float, halo: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(raw_path, allow_pickle=False) as saved:
        velocity = saved["velocity"].astype(np.float64)
    return (
        finite_difference_core(velocity, spacing, 8, halo),
        finite_difference_core(velocity, spacing, 6, halo),
    )


@st.cache_data(show_spinner=False)
def load_verification(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="计算逐帧质量指标……")
def frame_metrics(raw_path: str, derived_path: str, halo: int) -> pd.DataFrame:
    raw, derived = load_arrays(raw_path, derived_path)
    velocity = raw["velocity"].astype(np.float64)
    gradient8 = raw["gradient_primary"].astype(np.float64)
    gradient6 = raw["gradient_audit"].astype(np.float64)
    core = (slice(None), slice(None), slice(None)) + (slice(halo, -halo),) * 3
    g8 = gradient8[core]
    g6 = gradient6[core]
    axes_velocity = (1, 2, 3, 4)
    axes_gradient = (1, 2, 3, 4, 5)
    axes_scalar = (1, 2, 3)
    gradient_rms = np.sqrt(np.mean(g8 * g8, axis=axes_gradient))
    derivative_difference = np.sqrt(np.mean((g8 - g6) ** 2, axis=axes_gradient)) / gradient_rms
    divergence8 = np.trace(g8, axis1=1, axis2=2)
    divergence_ratio = np.sqrt(np.mean(divergence8**2, axis=axes_scalar)) / gradient_rms
    regime_disagreement = np.mean(derived["regime_primary"] != derived["regime_audit"], axis=axes_scalar)
    uncertain = np.mean(derived["regime_robust"] == 0, axis=axes_scalar)
    data: dict[str, Any] = {
        "frame": np.arange(len(raw["times"])),
        "time": raw["times"],
        "velocity_rms": np.sqrt(np.mean(velocity * velocity, axis=axes_velocity)),
        "gradient_fd8_rms": gradient_rms,
        "fd6_fd8_relative_rms": derivative_difference,
        "divergence_over_gradient_rms": divergence_ratio,
        "work_full_rms": np.sqrt(np.mean(derived["work_full"] ** 2, axis=axes_scalar)),
        "work_resolved_rms": np.sqrt(np.mean(derived["work_resolved"] ** 2, axis=axes_scalar)),
        "regime_disagreement": regime_disagreement,
        "uncertain_fraction": uncertain,
    }
    for code in range(1, 5):
        data[f"Q{code}"] = np.mean(derived["regime_robust"] == code, axis=axes_scalar)
    return pd.DataFrame(data)


def plane(field: np.ndarray, axis: str, index: int) -> tuple[np.ndarray, str, str]:
    if field.ndim != 3:
        raise ValueError(f"plot field must be 3-D; got {field.shape}")
    if axis == "z":
        return field[index, :, :], "x", "y"
    if axis == "y":
        return field[:, index, :], "x", "z"
    return field[:, :, index], "y", "z"


def heatmap(
    field: np.ndarray,
    title: str,
    axis: str,
    index: int,
    *,
    zmin: float | None = None,
    zmax: float | None = None,
    colorscale: str | list = "RdBu_r",
    colorbar_title: str = "value",
    regime: bool = False,
) -> go.Figure:
    values, xlabel, ylabel = plane(field, axis, index)
    if regime:
        integer_values = np.asarray(values, dtype=np.int64)
        labels = np.asarray(REGIME_LABELS, dtype=object)[np.clip(integer_values, 0, 4)]
        trace = go.Heatmap(
            z=values,
            customdata=labels,
            colorscale=REGIME_COLORS,
            zmin=-0.5,
            zmax=4.5,
            colorbar={"tickvals": [0, 1, 2, 3, 4], "ticktext": list(REGIME_LABELS)},
            hovertemplate=f"{xlabel}=%{{x}}<br>{ylabel}=%{{y}}<br>code=%{{z}}<br>%{{customdata}}<extra></extra>",
        )
    else:
        trace = go.Heatmap(
            z=values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar_title},
            hovertemplate=f"{xlabel}=%{{x}}<br>{ylabel}=%{{y}}<br>value=%{{z:.6g}}<extra></extra>",
        )
    figure = go.Figure(trace)
    figure.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 15}},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=360,
        margin={"l": 45, "r": 20, "t": 50, "b": 45},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    return figure


def show_comparison(
    entries: list[tuple[str, np.ndarray]],
    axis: str,
    index: int,
    *,
    difference_entries: set[int] | None = None,
    signed: bool = True,
) -> None:
    difference_entries = difference_entries or set()
    normal_planes = [plane(field, axis, index)[0] for n, (_, field) in enumerate(entries) if n not in difference_entries]
    if signed:
        normal_limit = max(float(np.max(np.abs(value))) for value in normal_planes) or 1.0
        normal_range = (-normal_limit, normal_limit)
        normal_scale = "RdBu_r"
    else:
        normal_range = (
            min(float(np.min(value)) for value in normal_planes),
            max(float(np.max(value)) for value in normal_planes),
        )
        normal_scale = "Viridis"
    for row_start in range(0, len(entries), 3):
        row = entries[row_start : row_start + 3]
        columns = st.columns(len(row))
        for local_index, (column, (name, field)) in enumerate(zip(columns, row)):
            global_index = row_start + local_index
            values = plane(field, axis, index)[0]
            if global_index in difference_entries:
                limit = float(np.max(np.abs(values))) or 1.0
                limits = (-limit, limit)
                scale = "RdBu_r"
            else:
                limits = normal_range
                scale = normal_scale
            column.plotly_chart(
                heatmap(field, name, axis, index, zmin=limits[0], zmax=limits[1], colorscale=scale),
                width="stretch",
            )


def direct_filter(field: np.ndarray, cfg: TaskConfig) -> np.ndarray:
    kernel = gaussian_kernel_1d(cfg.sigma_grid, cfg.support_radius)
    kernel3 = np.einsum("i,j,k->ijk", kernel, kernel, kernel)
    return convolve(np.asarray(field, dtype=np.float64), kernel3, mode="valid", method="direct")


def relative_rms(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denominator = float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float64) ** 2)))
    return float(np.sqrt(np.mean(difference**2)) / denominator) if denominator else float(np.sqrt(np.mean(difference**2)))


def frame_navigation(times: np.ndarray) -> int:
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0

    def move(delta: int) -> None:
        st.session_state.frame_index = int(np.clip(st.session_state.frame_index + delta, 0, len(times) - 1))

    previous, slider_column, following = st.columns([1, 6, 1])
    previous.button("← 上一帧", width="stretch", disabled=st.session_state.frame_index == 0, on_click=move, args=(-1,))
    following.button(
        "下一帧 →",
        width="stretch",
        disabled=st.session_state.frame_index == len(times) - 1,
        on_click=move,
        args=(1,),
    )
    slider_column.slider("帧", 0, len(times) - 1, key="frame_index")
    frame = int(st.session_state.frame_index)
    st.caption(f"frame={frame} / {len(times)-1}，physical time={times[frame]:.6f}")
    return frame


def overview_page(report: dict[str, Any], metrics: pd.DataFrame, frame: int) -> None:
    st.header("完整性与全局验证")
    checks = pd.DataFrame(
        [{"检查": name, "状态": "PASS" if passed else "FAIL"} for name, passed in report["checks"].items()]
    )
    st.dataframe(checks, width="stretch", hide_index=True)
    current = metrics.iloc[frame]
    columns = st.columns(4)
    columns[0].metric("当前帧 FD6/FD8", f"{current['fd6_fd8_relative_rms']:.3%}")
    columns[1].metric("div / grad RMS", f"{current['divergence_over_gradient_rms']:.3%}")
    columns[2].metric("regime 分歧", f"{current['regime_disagreement']:.3%}")
    columns[3].metric("uncertain", f"{current['uncertain_fraction']:.3%}")
    st.info(
        "这里的指标是同一位置误差或异常比例的空间汇总。单独的 velocity/gradient/work RMS "
        "只能描述量级，不能证明空间场正确；真正的检查应结合各页面的逐点差值图。"
    )
    st.subheader("跨帧质量指标")
    selected = st.multiselect(
        "曲线",
        [
            "fd6_fd8_relative_rms",
            "divergence_over_gradient_rms",
            "regime_disagreement",
            "uncertain_fraction",
        ],
        default=["fd6_fd8_relative_rms", "divergence_over_gradient_rms"],
    )
    if selected:
        long = metrics.melt(id_vars=["frame", "time"], value_vars=selected, var_name="metric", value_name="value")
        figure = px.line(long, x="time", y="value", color="metric", markers=False)
        figure.add_vline(x=float(current["time"]), line_dash="dash", line_color="black")
        st.plotly_chart(figure, width="stretch")
    with st.expander("验证限制", expanded=True):
        for limitation in report.get("limitations", []):
            st.write(f"- {limitation}")


def velocity_page(raw: dict[str, np.ndarray], derived: dict[str, np.ndarray], frame: int, cfg: TaskConfig, axis: str, index: int) -> None:
    st.header("速度场与滤波速度")
    component = COMPONENTS.index(st.selectbox("速度分量", COMPONENTS, format_func=lambda name: f"u_{name}"))
    h = cfg.support_radius
    raw_core = raw["velocity"][frame, component, h:-h, h:-h, h:-h]
    filtered = derived["velocity_bar"][frame, component]
    show_comparison(
        [("raw velocity（core）", raw_core), ("Gaussian filtered velocity", filtered), ("filtered − raw core", filtered - raw_core)],
        axis,
        index,
        difference_entries={2},
    )


def derivative_page(
    raw: dict[str, np.ndarray], local8: np.ndarray, local6: np.ndarray, frame: int, cfg: TaskConfig, axis: str, index: int
) -> None:
    st.header("数据库梯度与速度差分梯度")
    i = COMPONENTS.index(st.selectbox("速度分量 i", COMPONENTS, format_func=lambda name: f"u_{name}"))
    j = COMPONENTS.index(st.selectbox("求导方向 j", COMPONENTS, format_func=lambda name: f"∂/∂{name}"))
    h = cfg.support_radius
    db8 = raw["gradient_primary"][frame, i, j, h:-h, h:-h, h:-h]
    db6 = raw["gradient_audit"][frame, i, j, h:-h, h:-h, h:-h]
    l8 = local8[frame, i, j]
    l6 = local6[frame, i, j]
    columns = st.columns(3)
    columns[0].metric("DB FD8 vs local FD8", f"{relative_rms(db8, l8):.3e}")
    columns[1].metric("DB FD6 vs local FD6", f"{relative_rms(db6, l6):.3e}")
    columns[2].metric("DB FD6 vs FD8", f"{relative_rms(db8, db6):.3%}")
    show_comparison(
        [
            ("database FD8", db8),
            ("local FD8 from velocity", l8),
            ("local FD8 − database FD8", l8 - db8),
            ("database FD6", db6),
            ("local FD6 from velocity", l6),
            ("local FD6 − database FD6", l6 - db6),
            ("database FD8 − FD6", db8 - db6),
        ],
        axis,
        index,
        difference_entries={2, 5, 6},
    )


def filter_page(raw: dict[str, np.ndarray], derived: dict[str, np.ndarray], frame: int, cfg: TaskConfig, axis: str, index: int) -> None:
    st.header("生产滤波与独立直接三维卷积")
    field_kind = st.selectbox("被滤波场", ("velocity", "gradient", "acceleration"))
    i = COMPONENTS.index(st.selectbox("分量 i", COMPONENTS))
    if field_kind == "velocity":
        raw_field = raw["velocity"][frame, i]
        production = derived["velocity_bar"][frame, i]
    elif field_kind == "gradient":
        j = COMPONENTS.index(st.selectbox("导数方向 j", COMPONENTS))
        raw_field = raw["gradient_primary"][frame, i, j]
        production = derived["gradient_bar_primary"][frame, i, j]
    else:
        acceleration = advective_acceleration(raw["velocity"][frame : frame + 1], raw["gradient_primary"][frame : frame + 1])[0]
        raw_field = acceleration[i]
        production = derived["a_bar"][frame, i]
    independent = direct_filter(raw_field, cfg)
    h = cfg.support_radius
    raw_core = raw_field[h:-h, h:-h, h:-h]
    st.metric("direct 3-D vs production relative RMS", f"{relative_rms(production, independent):.3e}")
    show_comparison(
        [
            ("raw field（core）", raw_core),
            ("production separable filter", production),
            ("independent direct 3-D filter", independent),
            ("direct − production", independent - production),
        ],
        axis,
        index,
        difference_entries={3},
    )
    kernel = gaussian_kernel_1d(cfg.sigma_grid, cfg.support_radius)
    st.subheader("一维 Gaussian kernel")
    st.bar_chart(pd.DataFrame({"offset": np.arange(-cfg.support_radius, cfg.support_radius + 1), "weight": kernel}).set_index("offset"))
    st.caption(f"kernel sum={kernel.sum():.16f}; 3-D kernel 是三个一维 kernel 的外积。")


def acceleration_page(raw: dict[str, np.ndarray], derived: dict[str, np.ndarray], frame: int, cfg: TaskConfig, axis: str, index: int) -> None:
    st.header("Acceleration：FD8、FD6、a_bar 与 a_barbar")
    i = COMPONENTS.index(st.selectbox("acceleration 分量", COMPONENTS, format_func=lambda name: f"a_{name}"))
    velocity = raw["velocity"][frame : frame + 1].astype(np.float64)
    gradient8 = raw["gradient_primary"][frame : frame + 1].astype(np.float64)
    gradient6 = raw["gradient_audit"][frame : frame + 1].astype(np.float64)
    a8 = advective_acceleration(velocity, gradient8)[0]
    a6 = advective_acceleration(velocity, gradient6)[0]
    abar8 = derived["a_bar"][frame]
    abar6 = gaussian_valid(a6, cfg.sigma_grid, cfg.support_radius)
    vbar = derived["velocity_bar"][frame]
    gbar8 = derived["gradient_bar_primary"][frame]
    gbar6 = gaussian_valid(gradient6[0], cfg.sigma_grid, cfg.support_radius)
    abarbar8 = derived["a_barbar"][frame]
    abarbar6 = np.einsum("jzyx,ijzyx->izyx", vbar, gbar6, optimize=True)
    h = cfg.support_radius
    show_comparison(
        [
            ("raw acceleration FD8（core）", a8[i, h:-h, h:-h, h:-h]),
            ("raw acceleration FD6（core）", a6[i, h:-h, h:-h, h:-h]),
            ("raw FD8 − FD6", a8[i, h:-h, h:-h, h:-h] - a6[i, h:-h, h:-h, h:-h]),
            ("a_bar FD8", abar8[i]),
            ("a_bar FD6", abar6[i]),
            ("a_bar FD8 − FD6", abar8[i] - abar6[i]),
            ("a_barbar FD8", abarbar8[i]),
            ("a_barbar FD6", abarbar6[i]),
            ("a_barbar FD8 − FD6", abarbar8[i] - abarbar6[i]),
        ],
        axis,
        index,
        difference_entries={2, 5, 8},
    )


def work_regime_page(derived: dict[str, np.ndarray], frame: int, axis: str, index: int) -> None:
    st.header("Work、regime 与符号不稳定位置")
    wf8 = derived["work_full"][frame]
    wr8 = derived["work_resolved"][frame]
    wf6 = derived["work_full_audit"][frame]
    wr6 = derived["work_resolved_audit"][frame]
    show_comparison(
        [
            ("W_full FD8", wf8),
            ("W_full FD6", wf6),
            ("W_full FD8 − FD6", wf8 - wf6),
            ("W_resolved FD8", wr8),
            ("W_resolved FD6", wr6),
            ("W_resolved FD8 − FD6", wr8 - wr6),
        ],
        axis,
        index,
        difference_entries={2, 5},
    )
    primary = derived["regime_primary"][frame]
    audit = derived["regime_audit"][frame]
    robust = derived["regime_robust"][frame]
    disagreement = (primary != audit).astype(float)
    uncertain = (robust == 0).astype(float)
    st.subheader("五种 regime 分类")
    st.dataframe(
        pd.DataFrame(
            {
                "code": [0, 1, 2, 3, 4],
                "类别": list(REGIME_LABELS),
                "W_full": ["接近 0 或方法不稳定", "> 0", "> 0", "< 0", "< 0"],
                "W_resolved": ["接近 0 或方法不稳定", "> 0", "< 0", "> 0", "< 0"],
                "颜色": ["灰", "蓝", "橙", "绿", "红"],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    columns = st.columns(3)
    columns[0].plotly_chart(heatmap(primary, "regime FD8", axis, index, regime=True), width="stretch")
    columns[1].plotly_chart(heatmap(audit, "regime FD6", axis, index, regime=True), width="stretch")
    columns[2].plotly_chart(heatmap(robust, "robust regime", axis, index, regime=True), width="stretch")
    mask_columns = st.columns(2)
    mask_columns[0].plotly_chart(
        heatmap(disagreement, "FD6/FD8 disagreement mask", axis, index, zmin=0, zmax=1, colorscale="Magma"),
        width="stretch",
    )
    mask_columns[1].plotly_chart(
        heatmap(uncertain, "uncertain mask", axis, index, zmin=0, zmax=1, colorscale="Magma"),
        width="stretch",
    )
    st.caption(
        f"当前帧 disagreement={np.mean(disagreement):.4%}; uncertain={np.mean(uncertain):.4%}; "
        + ", ".join(f"Q{code}={np.mean(robust == code):.2%}" for code in range(1, 5))
    )


def divergence_page(raw: dict[str, np.ndarray], derived: dict[str, np.ndarray], frame: int, cfg: TaskConfig, axis: str, index: int) -> None:
    st.header("Divergence 与无散误差")
    div8 = derived["divergence_primary"][frame]
    div6 = derived["divergence_audit"][frame]
    divbar8 = derived["divergence_bar_primary"][frame]
    gradient6_bar = gaussian_valid(raw["gradient_audit"][frame], cfg.sigma_grid, cfg.support_radius)
    divbar6 = np.trace(gradient6_bar, axis1=0, axis2=1)
    show_comparison(
        [
            ("divergence FD8", div8),
            ("divergence FD6", div6),
            ("FD8 − FD6 divergence", div8 - div6),
            ("filtered divergence FD8", divbar8),
            ("filtered divergence FD6", divbar6),
            ("filtered FD8 − FD6", divbar8 - divbar6),
        ],
        axis,
        index,
        difference_entries={2, 5},
    )
    st.warning("数据库局部有限差分梯度不保证逐点 divergence=0。这里显示的是诊断量，程序不会静默投影或修改下载数据。")


def point_time_series_page(
    raw: dict[str, np.ndarray], derived: dict[str, np.ndarray], cfg: TaskConfig, frame: int
) -> None:
    st.header("选定 core 网格点的 100 帧局部时间序列")
    selector_columns = st.columns(5)
    ic = selector_columns[0].number_input("core i (x)", 0, cfg.core_shape[0] - 1, 4)
    jc = selector_columns[1].number_input("core j (y)", 0, cfg.core_shape[1] - 1, 4)
    kc = selector_columns[2].number_input("core k (z)", 0, cfg.core_shape[2] - 1, 4)
    component = COMPONENTS.index(selector_columns[3].selectbox("分量 i", COMPONENTS, index=0))
    derivative_axis = COMPONENTS.index(selector_columns[4].selectbox("求导方向 j", COMPONENTS, index=0))
    ic, jc, kc = int(ic), int(jc), int(kc)
    ib, jb, kb = ic + cfg.halo[0], jc + cfg.halo[1], kc + cfg.halo[2]
    global_i = cfg.block_start_ijk[0] + ib
    global_j = cfg.block_start_ijk[1] + jb
    global_k = cfg.block_start_ijk[2] + kb
    spacing = cfg.domain_length / cfg.grid_shape[0]
    st.caption(
        f"core=({ic},{jc},{kc}); block local=({ib},{jb},{kb}); global=({global_i},{global_j},{global_k}); "
        f"coordinate=({global_i*spacing:.6f},{global_j*spacing:.6f},{global_k*spacing:.6f})"
    )
    times = raw["times"]
    traces = pd.DataFrame(
        {
            "time": times,
            "velocity_raw": raw["velocity"][:, component, kb, jb, ib],
            "velocity_bar": derived["velocity_bar"][:, component, kc, jc, ic],
            "gradient_fd8": raw["gradient_primary"][:, component, derivative_axis, kb, jb, ib],
            "gradient_fd6": raw["gradient_audit"][:, component, derivative_axis, kb, jb, ib],
            "gradient_fd8_minus_fd6": raw["gradient_primary"][:, component, derivative_axis, kb, jb, ib]
            - raw["gradient_audit"][:, component, derivative_axis, kb, jb, ib],
            "a_bar": derived["a_bar"][:, component, kc, jc, ic],
            "a_barbar": derived["a_barbar"][:, component, kc, jc, ic],
            "work_full_fd8": derived["work_full"][:, kc, jc, ic],
            "work_resolved_fd8": derived["work_resolved"][:, kc, jc, ic],
            "work_full_fd6": derived["work_full_audit"][:, kc, jc, ic],
            "work_resolved_fd6": derived["work_resolved_audit"][:, kc, jc, ic],
            "divergence_fd8": derived["divergence_primary"][:, kc, jc, ic],
            "divergence_fd6": derived["divergence_audit"][:, kc, jc, ic],
            "regime_fd8": derived["regime_primary"][:, kc, jc, ic],
            "regime_fd6": derived["regime_audit"][:, kc, jc, ic],
            "regime_robust": derived["regime_robust"][:, kc, jc, ic],
        }
    )
    groups = {
        "速度": ["velocity_raw", "velocity_bar"],
        "梯度": ["gradient_fd8", "gradient_fd6", "gradient_fd8_minus_fd6"],
        "Acceleration": ["a_bar", "a_barbar"],
        "Work": ["work_full_fd8", "work_resolved_fd8", "work_full_fd6", "work_resolved_fd6"],
        "Divergence": ["divergence_fd8", "divergence_fd6"],
        "Regime": ["regime_fd8", "regime_fd6", "regime_robust"],
    }
    for group_name, names in groups.items():
        st.subheader(group_name)
        long = traces.melt(id_vars="time", value_vars=names, var_name="quantity", value_name="value")
        figure = px.line(long, x="time", y="value", color="quantity")
        figure.add_vline(x=float(times[frame]), line_dash="dash", line_color="black")
        if group_name == "Regime":
            figure.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=["uncertain", "Q1", "Q2", "Q3", "Q4"])
        st.plotly_chart(figure, width="stretch")
    st.dataframe(traces, width="stretch", hide_index=True)


def time_series_page(metrics: pd.DataFrame, frame: int) -> None:
    st.header("100 帧时间序列")
    validation_options = [
        "fd6_fd8_relative_rms",
        "divergence_over_gradient_rms",
        "regime_disagreement",
        "uncertain_fraction",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]
    selected = st.multiselect(
        "验证指标（同一位置误差、异常比例或 regime occupancy）",
        validation_options,
        default=["fd6_fd8_relative_rms", "divergence_over_gradient_rms", "regime_disagreement"],
    )
    if selected:
        normalized = st.checkbox("每条曲线除以自身 RMS（便于放在同一图中）", value=False)
        plot_data = metrics[["frame", "time"] + selected].copy()
        if normalized:
            for name in selected:
                scale = float(np.sqrt(np.mean(plot_data[name] ** 2))) or 1.0
                plot_data[name] /= scale
        long = plot_data.melt(id_vars=["frame", "time"], value_vars=selected, var_name="metric", value_name="value")
        figure = px.line(long, x="time", y="value", color="metric")
        figure.add_vline(x=float(metrics.iloc[frame]["time"]), line_dash="dash", line_color="black")
        st.plotly_chart(figure, width="stretch")
    with st.expander("空间量级汇总（不作为场正确性的判据）"):
        amplitude_options = ["velocity_rms", "gradient_fd8_rms", "work_full_rms", "work_resolved_rms"]
        amplitude_selected = st.multiselect("量级曲线", amplitude_options, default=[])
        if amplitude_selected:
            long = metrics.melt(
                id_vars=["frame", "time"], value_vars=amplitude_selected, var_name="metric", value_name="value"
            )
            figure = px.line(long, x="time", y="value", color="metric")
            figure.add_vline(x=float(metrics.iloc[frame]["time"]), line_dash="dash", line_color="black")
            st.plotly_chart(figure, width="stretch")
        st.caption(
            "这些 RMS 对空间点和分量取平均，只用于发现整体量级突变或时间漂移；它们会丢失空间结构，Work RMS 还会丢失符号。"
        )
    st.dataframe(metrics, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="JHTDB Task 0 验证", page_icon="🌊", layout="wide")
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_config(project_root / "configs" / "task0.yaml")
    verification_path = cfg.reports_path / "task0_verification.json"
    required = (cfg.raw_path, cfg.derived_path, verification_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        st.error("缺少数据文件：\n" + "\n".join(missing))
        st.code("python task0.py compute\npython task0.py verify")
        st.stop()
    raw, derived = load_arrays(str(cfg.raw_path), str(cfg.derived_path))
    report = load_verification(str(verification_path))
    spacing = cfg.domain_length / cfg.grid_shape[0]
    local8, local6 = reconstruct_gradients(str(cfg.raw_path), spacing, cfg.support_radius)
    metrics = frame_metrics(str(cfg.raw_path), str(cfg.derived_path), cfg.support_radius)

    st.title("JHTDB Task 0：逐帧验证面板")
    status_color = "green" if report.get("overall") == "PASS" else "red"
    st.markdown(f"总体离线验证：<span style='color:{status_color};font-weight:700'>{report.get('overall')}</span>", unsafe_allow_html=True)
    frame = frame_navigation(raw["times"])

    with st.sidebar:
        st.header("显示控制")
        page = st.radio(
            "验证页面",
            (
                "总览与完整性",
                "速度场",
                "导数对比",
                "滤波对比",
                "Acceleration",
                "Work 与 regimes",
                "Divergence",
                "单点随时间",
                "跨帧时间序列",
            ),
        )
        axis = st.selectbox("切片法向", ("z", "y", "x"), format_func=lambda value: f"{value}=constant")
        slice_index = st.slider("core 切片 index", 0, cfg.core_shape[0] - 1, cfg.core_shape[0] // 2)
        st.divider()
        st.write(f"raw: `{cfg.raw_path.name}`")
        st.write(f"derived: `{cfg.derived_path.name}`")
        st.write(f"block/core: `{cfg.block_shape}` / `{cfg.core_shape}`")
        st.write("所有页面均为离线读取，不访问 JHTDB。")

    if page == "总览与完整性":
        overview_page(report, metrics, frame)
    elif page == "速度场":
        velocity_page(raw, derived, frame, cfg, axis, slice_index)
    elif page == "导数对比":
        derivative_page(raw, local8, local6, frame, cfg, axis, slice_index)
    elif page == "滤波对比":
        filter_page(raw, derived, frame, cfg, axis, slice_index)
    elif page == "Acceleration":
        acceleration_page(raw, derived, frame, cfg, axis, slice_index)
    elif page == "Work 与 regimes":
        work_regime_page(derived, frame, axis, slice_index)
    elif page == "Divergence":
        divergence_page(raw, derived, frame, cfg, axis, slice_index)
    elif page == "单点随时间":
        point_time_series_page(raw, derived, cfg, frame)
    else:
        time_series_page(metrics, frame)
