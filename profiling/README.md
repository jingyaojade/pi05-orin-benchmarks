# 独立内存补测

本目录的 `memory.py` 是**内存专用 pass**。正式延迟继续使用原有
`baseline/benchmark.py` 和 `flashrt/benchmark.py`；这里的阶段同步、读取
`/proc`、`sudo -n cat` 和采样线程都会扰动执行时间，因此不输出可比较的推理延迟。
脚本不停止服务、不更改功耗/时钟、不更改模型或原 benchmark 源文件。
由运行者先确保其他模型服务不会与本次测量争用 GPU，并在结束后恢复服务。

每个 backend 都必须在相应 Python 环境下启动**全新的进程**；不得在同一进程先后加载两个引擎。
使用与正式延迟试验相同的 `fixed_inputs.npz` 和 `reference_actions.npy`。
默认 warmup 5 次，内存重复 3 次。阶段结果保持与本 backend 第一份输出逐元素一致；
baseline 同时要求与保存的 baseline reference 一致。FlashRT 单独报告相对 baseline 的误差，
不要求不同数值精度跨引擎 bitwise 相同。

```bash
# 从仓库根目录运行；根据实际 baseline 输出目录调整 fixture。
fixture=reports/baseline-2026-10-16/fixed_inputs.npz

JAX_PLATFORMS=cpu OMP_NUM_THREADS=6 \
  /home/nvidia/pi05-orin/.venv/bin/python profiling/memory.py \
  --backend baseline --deployment-root /home/nvidia/pi05-orin \
  --fixture "$fixture" --output reports/memory-baseline

OMP_NUM_THREADS=6 \
  /home/nvidia/pi05-orin-flashrt/.venv/bin/python profiling/memory.py \
  --backend flashrt --flashrt-root /home/nvidia/pi05-orin-flashrt/vendor/FlashRT \
  --checkpoint /home/nvidia/pi05-orin/models/pi05_libero \
  --precision bf16 --fixture "$fixture" --output reports/memory-flashrt-bf16

# INT8 是数值误差更大的独立实验；不能用 BF16 名称引用它。
OMP_NUM_THREADS=6 \
  /home/nvidia/pi05-orin-flashrt/.venv/bin/python profiling/memory.py \
  --backend flashrt --flashrt-root /home/nvidia/pi05-orin-flashrt/vendor/FlashRT \
  --checkpoint /home/nvidia/pi05-orin/models/pi05_libero \
  --precision int8 --fixture "$fixture" --output reports/memory-flashrt-int8
```

首次启动自动尝试直接读取 `/sys/kernel/debug/nvmap/iovmm/clients`，失败后尝试
`sudo -n cat`。后者依赖现有免交互 sudo 权限，脚本不会提示输入密码或修改 sudo 配置。
如果该接口不可访问，保存原因并将 nvmap 数据留空；可用 `--disable-nvmap` 明确禁用。
`--help` 和 CPU 单元检查不需要 CUDA、Torch 或模型文件。

## 指标的边界

| 指标 | 来源与解释 |
|---|---|
| `process.rss_bytes` | `/proc/self/smaps_rollup` 的 Rss，缺失时回退 status 的 VmRSS。进程主机映射驻留页。 |
| `process.pss_bytes` | smaps_rollup 的 Pss，将共享映射按进程分摊。 |
| `process.uss_bytes` | Private_Clean + Private_Dirty。保留完整 smaps 原文用于核对；未将 Private_Hugetlb 偷加到常规 USS 定义中。 |
| `process.lifetime_rss_high_watermark_bytes` | status 的 VmHWM，**进程生命周期** RSS 高水位，不能当成特定模型阶段峰值。 |
| `nvmap.self_bytes` | Jetson debugfs clients 表中当前 Python PID 的所有 client SIZE 之和，K 按 1024 字节。比全设备余量更可归属，但不等于独占物理 DRAM。保留整张原表。 |
| `torch.allocated/reserved/peak_*_bytes` | Torch caching allocator。peak reset 之后精确跟踪该 allocator，**不能覆盖 FlashRT 原生 cudaMalloc、图内部或库内部全部内存**。 |
| `cuda_device.used/free/total_bytes` | cudaMemGetInfo 的全设备范围，包含其他进程和驱动影响。不能将变化量称为当前模型精确分配量。 |
| `cuda_buffer_inventory.json` | FlashRT frontend 可达的持久 CudaBuffer 对象，按实际 base pointer 去重，并记录 nbytes、managed 属性和全部别名。排除 Torch allocator 与不由 CudaBuffer 管理的库/图内存。 |

**Jetson 使用统一内存。RSS/PSS/USS、nvmap 和 CUDA 全设备用量的记账域可能重叠，
任何两列都不能直接相加为“总内存”。** 建议报告并列展示 nvmap 当前 PID、RSS/PSS、
Torch live/reserved，并注明各自边界。不要把“全设备 used 的差值”当成精确进程峰值。

阶段前后都同步整个设备后读取快照；每次 `measure()` 在阶段开始重置 Torch peak。
只包装不重叠的阶段，不在 model_total 的内部再次重置 peak：

1. 独立 model_total pass：CUDA 输入已就绪，原始模型/原始生产图运行至完成。
   包含未被内部 reset 打断的 Torch allocator peak。后台轮询等待间隔默认 100 ms；
   读取本身另需约 40–52 ms，因此真实采样间隔更长，不能解释为精确 10 Hz。
2. 独立 stage pass：baseline 包装原实例的 model_preprocess、3 次 vision、language embedding、
   prefill、10 次 denoise；CPU preprocess、H2D+Observation、D2H+postprocess 由外层测量。
   FlashRT 使用现有独立诊断 vision/prefill/denoise 图，并单列 input_prepare_and_staging、D2H+postprocess。
   FlashRT 原 frontend 的 CPU 转换和上传复合边界保持原样，不捏造 CPU-only 与 H2D 的拆分。
3. backend_load 和 FlashRT prepare/capture 也有独立采样，记录加载过程观察到的内存高值。

`torch_peak_allocated_bytes` 是该区域的**绝对 allocator 峰值**；
`torch_peak_above_start_bytes` 是该峰值高于区域开始 allocated 的非负部分；
`endpoint_delta_bytes` 只是结束减开始的变化，不能当成峰值。
`max_observed_bytes` 是同步边界及周期采样中观察到的最大值，**仅是瞬时峰值的下界**，
尤其不能据此宣称捕获了全部原生 CUDA 短暂分配。
细分 vision、prefill、denoise 等阶段仅有进程/nvmap 同步前后边界数据；
加载、prepare/capture 和 model_total 才启用后台轮询。Torch allocator peak 由计数器捕获。

模型权重、KV、scratch 和 CUDA Graph 缓冲区通常跨阶段常驻。阶段前后相同、阶段增量接近零是
预分配推理的正常现象；每阶段所见的常驻占用不能累计。FlashRT 诊断图在生产 model_total pass
结束后才捕获，因此正式生产图的稳态快照不会混入这些额外诊断图对象。

## 结果文件与 schema

所有数字内存单位为 bytes；JSON 不将 MiB 与 bytes 混用。

- `snapshots.jsonl`：按 snapshot_id 排序可复原全部原始快照。每条含 UTC、monotonic_ns、PID、
  `/proc` 原文与解析字段、nvmap 原表与同 PID client 列表；边界快照另有 Torch/CUDA 指标。
  后台采样仅读取主机/proc/nvmap，不从采样线程调用 CUDA 或同步 GPU。
- `raw.json`：`schema_version`、`checkpoints[]`、`measurements[]`、`metadata`。
  每条 measurement 包括 stage、sample、iteration、before、after、sampled_host_snapshot_ids、
  max_observed_bytes、Torch 峰值和起点以上增量、endpoint_delta_bytes。
- `samples.csv`：每阶段/每次调用一行；before_/after_/max_observed_ 展开为独立列，便于交叉核对。
- `summary.json`：各 stage 的重复样本 mean/min/max，以及命名 checkpoint、测量元数据。
  baseline vision 含 3 次调用、denoise_iteration 含 10 次调用，iteration 详情留在 raw/CSV；不可把它们加权成一轮峰值。
- `provenance.json`：输入与 reference SHA256、脚本 SHA256、Python/Torch/CUDA/GPU、checkpoint 路径、
  FlashRT pin/补丁/tokenizer、动作一致性校验、限制与成功/失败状态。重量文件 SHA 复用同 fixture 的正式 latency provenance。
- `cuda_buffer_inventory.json`：仅 FlashRT，分别记录生产图 warmup 后与诊断图捕获后的持久 CudaBuffer 清单。

文件不能覆盖已有非空输出目录。异常发生时尽力保存已收集快照和失败原因。
采样和 JSON 记账自身也占主机内存，因此所有数值都包含 profiler 的开销。

CPU 检查：

```bash
python -m unittest discover -s profiling/tests -v
python profiling/memory.py --help
```

## 完整重测与报告

从仓库根目录执行 `bash profiling/run_latency.sh reports/new-round`，随后执行
`bash profiling/run_profiles.sh reports/new-round`。这些外层脚本每轮只暂停/恢复一次
`openpi-pi05`，会短暂停止该服务的请求；其他进程保持运行。恢复时仅在服务返回
`start-limit-hit` 时清除该服务的失败/启动频率计数，不改 systemd 单元配置。
已存在的非空结果目录不能复用。低层 Python profiler 自身不管理服务。

`pipeline_latency.py` 是 FlashRT 输入准备+上传、原始完整图、下载+后处理的独立同步诊断；
输入包括 CPU RGB 转换与 noise/image H2D/D2D，未伪造独立 CPU/H2D 数字。
额外同步使它与原 production policy 的采样边界不同；正式延迟仍来自原 benchmark。

更新 manifest 中的目录后，用 `python reporting/engine_comparison.py --manifest
reports/engine-comparison-2026-10-16/manifest.json --output reports/new-comparison.md`
生成 Markdown 与同名 CSV；缺失引擎指标保持 N/A/空值。
