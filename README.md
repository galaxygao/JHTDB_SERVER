# JHTDB SciServer 全周期域流水线

本项目在 Johns Hopkins SciServer 内完成 `isotropic1024coarse` 单帧速度场获取、全周期域谱计算、中心 `512^3` 裁剪、结果验证和服务器端可视化。科学数据不与本地电脑交互。

详细的平台结构、容器创建、配置和操作说明见 [`SCISERVER_SYSTEM_GUIDE.md`](SCISERVER_SYSTEM_GUIDE.md)。

## 1. 运行架构

```text
JHTDB / Turbulence (ceph)
        │ 128^3 tile，严格串行
        ▼
scratch：完整 1024^3 × 3 速度缓存
        │ 全周期域谱导数与谱高斯滤波
        ▼
persistent：results/.staging/<run_id>
        │ 中心 [256:768)^3 校验
        ▼
persistent：results/<run_id> + COMPLETE
        │
        └── Interactive container 中的只读 GUI
```

平台角色：

- Interactive Compute container：环境准备、`doctor`、测试、`smoke` 和服务器 GUI；
- shell-command Compute Job：当前用于正式单帧计算；多尺度流程在首帧验收后另行实现；
- `Turbulence (ceph)`：JHTDB 数据访问；
- `scratch`：全域速度、FFT/memmap 和其他可重建中间量；
- `persistent`：代码、环境、状态、QA、manifest 和正式中心结果。

当前账户 Quotas 页面在 2026-08-28 显示 `Storage on FileServiceJHU` 为 **100 GB**，因此本项目按该账户级配额设计；旧文档中的 10 GB 默认值不适用于当前账户。SciServer 没有在项目内提供可靠的账户配额 API，所以 `doctor` 会报告配置中记录的 100 GB，并实测挂载文件系统剩余空间；提交任务前仍以 Quotas 页面为最终依据。项目默认保留至少 15 GiB persistent 安全余量。

`Temporary` 的 72 小时生命周期、无备份属性和容器层非持久性仍按随项目保存的 [`Policies – SciServer.pdf`](Policies%20%E2%80%93%20SciServer.pdf) 执行。

## 2. 科学数据与输出

单帧输入：

- dataset：`isotropic1024coarse`；
- variable：`velocity`；
- 完整网格：`1024^3`；
- 周期域：`[0, 2π)^3`；
- 内部轴顺序：`(component, z, y, x)`；
- JHTDB tile：`128^3`，同一时刻只允许一个请求在途。

所有 FFT、谱导数和谱滤波必须先在完整 `1024^3` 周期域完成。之后才提取三个轴相同的半开区间：

```text
x = [256, 768)
y = [256, 768)
z = [256, 768)
shape = (512, 512, 512)
```

正式 persistent 结果包含：

| 字段 | shape | dtype | 未压缩大小 |
|---|---:|---:|---:|
| `velocity` | `(3,512,512,512)` | `float32` | 1.5 GiB |
| `gradient` | `(3,3,512,512,512)` | `float32` | 4.5 GiB |
| `velocity_bar` | `(3,512,512,512)` | `float32` | 1.5 GiB |
| `gradient_bar` | `(3,3,512,512,512)` | `float32` | 4.5 GiB |
| `work_full` | `(512,512,512)` | `float32` | 0.5 GiB |
| `work_resolved` | `(512,512,512)` | `float32` | 0.5 GiB |
| `regime` | `(512,512,512)` | `uint8` | 0.125 GiB |

单尺度合计约 13.125 GiB，另有 Zarr metadata 和文件系统开销。正式结果不打包、不分卷、不下载到本地。

## 3. 分块含义

项目只保留两种服务器内部 chunk：

1. JHTDB `128^3` tile：限制单次请求、支持串行获取和断点恢复；
2. Zarr chunk：支持增量写入、回读校验和 GUI 二维切片。

它们不是传输分卷。项目不生成 `tar.zst`、`part-*` 或本地归档。

## 4. SciServer 环境

建议选择：

| 项目 | 选择 |
|---|---|
| Compute Image | `SciServer Essentials 4.0` |
| Data Volume | `Turbulence (ceph)` |
| User Volume | `persistent`，读写 |
| User Volume | `scratch`，读写 |
| Python | 3.10 或更高 |

项目路径：

```text
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
```

大型运行目录：

```text
/home/idies/workspace/Temporary/gaoxingqun/scratch/JHTDB_RUNS
```

正式结果目录：

```text
/home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA/results
```

## 5. 安装和认证

在交互容器的 JupyterLab Terminal 中：

```bash
cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA
bash scripts/bootstrap.sh
source .venv/bin/activate
```

JHTDB token 仅从以下位置读取：

1. 当前进程环境变量 `JHTDB_TOKEN`；
2. 配置指定的、项目目录之外且权限为 `0600` 的 token 文件。

token 不得进入 YAML、Git、CLI 参数、日志、QA、manifest 或 GUI。

## 6. 运行前检查

```bash
python -m jhtdb_pipeline doctor --config configs/pipeline.yaml
python -m unittest discover -s tests -v
python -m jhtdb_pipeline plan --time-index 1 --config configs/pipeline.yaml
python -m jhtdb_pipeline smoke --time-index 1 --config configs/pipeline.yaml
```

- `doctor`：检查 SciServer、镜像、挂载卷、token 状态、空间和 scratch 到期时间；
- `unittest`：只运行离线测试，不访问 JHTDB；
- `plan`：只生成请求与空间计划；
- `smoke`：读取一个 `8^3` 真实小块。

完整单帧任务只有在以上检查通过后才允许启动。

## 7. 正式单帧 Compute Job

Jobs 页面选择相同 image、`Turbulence (ceph)`、可写 persistent 和可写 scratch，然后提交：

```bash
bash -lc 'cd /home/idies/workspace/Storage/gaoxingqun/persistent/JHU_DATA && source .venv/bin/activate && bash scripts/run_stage.sh single-frame --time-index 1 --sigma-grid 1.0'
```

正常执行顺序：

```text
doctor → cache → validate-input → process-center → finalize-result → status
```

- `cache` 可按 tile 断点续跑；
- `process-center` 只把裁剪后的最终字段写入 persistent staging；
- `finalize-result` 完整验证后才创建正式结果和 `COMPLETE`；
- staging 不完整时不得由 GUI 当作正式数据；
- job 完成并确认正式结果后，可删除 scratch 中间量。

## 8. 服务器 GUI

GUI 只读取 persistent 中存在 `COMPLETE` 标记的正式结果：

```bash
python -m jhtdb_pipeline gui --config configs/pipeline.yaml
```

它按需读取二维切片，支持：

- `velocity` 与 `velocity_bar` 对比；
- 9 个 `gradient` 与 `gradient_bar` 对比；
- 线性或 SymLog 梯度色标；
- `work_full`、`work_resolved` 和 `regime`；
- QA、manifest 和尺度信息。

该命令监听 `0.0.0.0:8501`。需要在交互容器的端口入口中打开；若当前 compute domain 不提供端口代理，则需在服务器端补充 Jupyter 内嵌 viewer 后再可视化，不回退到本地 GUI。

## 9. 后续多尺度任务

当前代码和 `run_stage.sh` **只接受单帧、单尺度**。首帧验收后再实现多尺度 job；届时公共 `velocity` 和 `gradient` 只保存一次，每个尺度分别保存 `velocity_bar`、`gradient_bar`、work 和 regime。启动前必须根据 persistent 实时余量计算可容纳的尺度数量。

## 10. 项目结构

```text
JHU_DATA/
├── configs/pipeline.yaml
├── scripts/
│   ├── bootstrap.sh
│   └── run_stage.sh
├── src/jhtdb_pipeline/
├── tests/
├── dashboard.py
├── pyproject.toml
├── README.md
├── SCISERVER_SYSTEM_GUIDE.md
└── Policies – SciServer.pdf
```

不保留 Windows 启动器、`givernylocal`、keyring、本地路径、本地下载器、结果分卷器或旧版全域派生数据格式。

## 11. 正确性门槛

- 输入 tile 覆盖完整、无重叠、SHA-256 回读一致；
- Giverny `(z,y,x,component)` 明确转换为 `(component,z,y,x)`；
- 所有输入和输出均检查 shape、dtype 与有限值；
- 全域无散度验证通过；
- 禁止“先裁剪、后 FFT”；
- 中心 raw/filtered 字段使用完全相同的坐标与切片；
- persistent staging 全部验证成功后才能原子提升；
- token 不出现在任何运行产物；
- 普通测试和安装不得触发完整真实数据任务。

## 12. 官方资料

- JHTDB `giverny`：<https://github.com/sciserver/giverny>
- SciServer Compute：<https://apps.sciserver.org/compute/>
- SciServer Python/Jobs API：<https://www.sciserver.org/docs/sciscript-python/SciServer.html>
