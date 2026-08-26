from __future__ import annotations

import json
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import zarr

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config


REGIME_LABELS = ("uncertain", "Q1: +/+", "Q2: +/-", "Q3: -/+", "Q4: -/-")
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


def extract_slice(array, component: int, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return np.asarray(array[component, :, :, index])
    if axis == "y":
        return np.asarray(array[component, :, index, :])
    if axis == "z":
        return np.asarray(array[component, index, :, :])
    raise ValueError(f"unknown axis {axis}")


def open_snapshot_readonly(raw_store_path, time_index: int):
    root = zarr.open_group(str(raw_store_path), mode="r")
    return root[f"t{time_index:06d}"]["velocity"]


def open_derived_readonly(derived_store_path, time_index: int):
    root = zarr.open_group(str(derived_store_path), mode="r")
    return root[f"t{time_index:06d}"]


def open_gradient_readonly(gradient_store_path, time_index: int):
    root = zarr.open_group(str(gradient_store_path), mode="r")
    return root[f"t{time_index:06d}"]


def open_filtered_readonly(filtered_store_path, time_index: int):
    root = zarr.open_group(str(filtered_store_path), mode="r")
    return root[f"t{time_index:06d}"]


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


def _continuous_figure(values: np.ndarray, title: str, *, signed: bool = True):
    stride = max(1, values.shape[0] // 512)
    shown = values[::stride, ::stride]
    kwargs = {}
    if signed:
        finite = shown[np.isfinite(shown)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        limit = limit or 1.0
        kwargs = {"zmin": -limit, "zmax": limit}
    figure = px.imshow(
        shown,
        origin="lower",
        color_continuous_scale="RdBu_r" if signed else "Viridis",
        aspect="equal",
        **kwargs,
    )
    figure.update_layout(title=title)
    return figure


def _regime_figure(values: np.ndarray, title: str):
    stride = max(1, values.shape[0] // 512)
    shown = np.asarray(values[::stride, ::stride], dtype=np.uint8)
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
    st.set_page_config(page_title="JHTDB full-domain viewer", layout="wide")
    config_path = os.environ.get("JHTDB_PIPELINE_CONFIG", "configs/pipeline.yaml")
    cfg = load_config(config_path)
    st.title("JHTDB 全周期域速度场（只读）")
    st.caption(f"Data root: {cfg.storage_root}")
    if not cfg.catalog_path.exists() or not cfg.raw_store_path.exists():
        st.info("尚无已下载数据。")
        return
    with Catalog(cfg.catalog_path) as catalog:
        snapshots = catalog.snapshots(cfg.dataset)
        if not snapshots:
            st.info("catalog 中没有快照。")
            return
        labels = [f"{row['time_index']} | t={row['physical_time']:.6f} | {row['status']}" for row in snapshots]
        selected = st.sidebar.selectbox("Snapshot", range(len(labels)), format_func=lambda i: labels[i])
        row = snapshots[selected]
        time_index = int(row["time_index"])
        tiles = catalog.tiles(cfg.dataset, time_index)
    st.subheader("状态")
    verified = sum(tile["status"] == "verified" for tile in tiles)
    st.write({"time_index": time_index, "status": row["status"], "verified_tiles": f"{verified}/{len(tiles)}"})

    page = st.sidebar.radio(
        "页面",
        ("原始速度", "速度梯度", "滤波速度与梯度", "Work 与 regime", "质量报告"),
    )
    axis = st.sidebar.selectbox("切片法向", ["z", "y", "x"])
    index = st.sidebar.slider("切片 index", 0, cfg.grid_shape[0] - 1, cfg.grid_shape[0] // 2)

    if page == "原始速度":
        array = open_snapshot_readonly(cfg.raw_store_path, time_index)
        component = st.sidebar.selectbox("速度分量", [0, 1, 2], format_func=lambda i: ["ux", "uy", "uz"][i])
        plane = extract_slice(array, component, axis, index)
        st.plotly_chart(
            _continuous_figure(plane, f"{['ux','uy','uz'][component]} · {axis}={index}"),
            use_container_width=True,
        )
        if row["status"] != "auto_validated":
            st.warning("该快照尚未完成自动验证；未下载区域可能显示为空白。")

    elif page == "速度梯度":
        if not cfg.gradient_store_path.exists():
            st.info("尚无托管梯度。先运行 gradient。")
        else:
            try:
                gradient_group = open_gradient_readonly(
                    cfg.gradient_store_path, time_index
                )
            except KeyError:
                st.info("当前时间帧尚无托管梯度。")
            else:
                if gradient_group.attrs.get("status") != "complete":
                    st.warning("梯度尚未全部完成并验证，暂不绘图。")
                else:
                    velocity_component = st.sidebar.selectbox(
                        "速度分量 i", [0, 1, 2], format_func=lambda i: ["ux", "uy", "uz"][i]
                    )
                    derivative_component = st.sidebar.selectbox(
                        "求导方向 j", [0, 1, 2], format_func=lambda j: ["x", "y", "z"][j]
                    )
                    gradient = gradient_group["gradient"]
                    plane = extract_gradient_slice(
                        gradient,
                        velocity_component,
                        derivative_component,
                        axis,
                        index,
                    )
                    title = (
                        f"∂u{['x','y','z'][velocity_component]}"
                        f"/∂{['x','y','z'][derivative_component]} · {axis}={index}"
                    )
                    st.plotly_chart(
                        _continuous_figure(plane, title), use_container_width=True
                    )
                    st.caption(
                        f"gradient manifest: {gradient_group.attrs.get('manifest_hash', 'missing')}"
                    )

    elif page == "滤波速度与梯度":
        if not cfg.filtered_store_path.exists():
            st.info("尚无完成的滤波速度与梯度。先运行 gradient。")
        else:
            try:
                filtered = open_filtered_readonly(cfg.filtered_store_path, time_index)
            except KeyError:
                st.info("当前时间帧尚无滤波速度与梯度。")
            else:
                if filtered.attrs.get("status") != "complete":
                    st.warning("滤波预处理尚未完成并验证，暂不绘图。")
                else:
                    field = st.sidebar.radio("滤波场", ("velocity_bar", "gradient_bar"))
                    velocity_component = st.sidebar.selectbox(
                        "速度分量 i",
                        [0, 1, 2],
                        format_func=lambda i: ["ux", "uy", "uz"][i],
                    )
                    if field == "velocity_bar":
                        plane = extract_slice(
                            filtered["velocity_bar"], velocity_component, axis, index
                        )
                        title = (
                            f"filtered u{['x','y','z'][velocity_component]} · {axis}={index}"
                        )
                    else:
                        derivative_component = st.sidebar.selectbox(
                            "求导方向 j",
                            [0, 1, 2],
                            format_func=lambda j: ["x", "y", "z"][j],
                        )
                        plane = extract_gradient_slice(
                            filtered["gradient_bar"],
                            velocity_component,
                            derivative_component,
                            axis,
                            index,
                        )
                        title = (
                            f"∂filtered u{['x','y','z'][velocity_component]}"
                            f"/∂{['x','y','z'][derivative_component]} · {axis}={index}"
                        )
                    st.plotly_chart(
                        _continuous_figure(plane, title), use_container_width=True
                    )
                    st.caption(
                        f"filtered manifest: {filtered.attrs.get('manifest_hash', 'missing')}"
                    )

    elif page == "Work 与 regime":
        if not cfg.derived_store_path.exists():
            st.info("尚无物理计算结果。先运行 compute。")
        else:
            try:
                derived = open_derived_readonly(cfg.derived_store_path, time_index)
            except KeyError:
                st.info("当前时间帧尚无物理计算结果。")
            else:
                derived_status = derived.attrs.get("status", "incomplete")
                if derived_status != "complete":
                    st.warning(f"物理结果状态为 {derived_status}；为避免展示半成品，暂不绘图。")
                else:
                    full_plane = extract_scalar_slice(derived["work_full"], axis, index)
                    resolved_plane = extract_scalar_slice(derived["work_resolved"], axis, index)
                    regime_plane = extract_scalar_slice(derived["regime"], axis, index)
                    left, right = st.columns(2)
                    left.plotly_chart(
                        _continuous_figure(full_plane, f"work_full · {axis}={index}"),
                        use_container_width=True,
                    )
                    right.plotly_chart(
                        _continuous_figure(resolved_plane, f"work_resolved · {axis}={index}"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        _regime_figure(regime_plane, f"regime · {axis}={index}"),
                        use_container_width=True,
                    )
                    metrics = st.columns(2)
                    metrics[0].metric("epsilon_full", f"{float(derived.attrs['epsilon_full']):.6g}")
                    metrics[1].metric("epsilon_resolved", f"{float(derived.attrs['epsilon_resolved']):.6g}")
                    occupancy = dict(derived.attrs.get("occupancy", {}))
                    if occupancy:
                        st.subheader("全域 regime 占比")
                        st.bar_chart(occupancy)

    else:
        st.subheader("原始数据 QA")
        qa_file = cfg.qa_path / f"t{time_index:06d}.json"
        if qa_file.exists():
            st.json(json.loads(qa_file.read_text(encoding="utf-8")))
        else:
            st.info("原始数据 QA 尚未生成。")
        st.subheader("梯度 FD8 审计")
        gradient_audit_file = cfg.qa_path / f"gradient_audit_t{time_index:06d}.json"
        if gradient_audit_file.exists():
            st.json(json.loads(gradient_audit_file.read_text(encoding="utf-8")))
        else:
            st.info("梯度 FD8 审计尚未生成。")
        st.subheader("全域无散性")
        divergence_file = cfg.qa_path / f"divergence_t{time_index:06d}.json"
        if divergence_file.exists():
            st.json(json.loads(divergence_file.read_text(encoding="utf-8")))
        else:
            st.info("全域无散性报告尚未生成。")
        st.subheader("物理计算报告")
        physics_file = cfg.qa_path / f"physics_t{time_index:06d}.json"
        if physics_file.exists():
            st.json(json.loads(physics_file.read_text(encoding="utf-8")))
        else:
            st.info("物理计算报告尚未生成。")

    st.caption("本 GUI 不写入 raw 数据、不保存 accept/reject，也不控制物理计算。")


if __name__ == "__main__":
    main()
