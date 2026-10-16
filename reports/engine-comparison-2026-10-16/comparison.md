# π0.5 Orin 推理引擎实测对比

数值全部由 manifest 指向的原始结果生成；N/A 表示未测量或未获得可比较结果，不能解释为零。延迟单位 ms，内存单位 MiB（1 MiB = 1,048,576 bytes；1 GiB = 1,024 MiB）。

## 原始模型与完整 policy 延迟

模型边界为 CUDA 输入已就绪至 CUDA 输出完成，不含输入上传、D2H 与动作反归一化。完整 policy 包含服务进程内前后处理，未包含网络通信；FlashRT 的固定 prompt 已预先缓存，baseline public policy 每请求处理 prompt。加速比以本 manifest 首个可运行 baseline 的模型 mean 为比较基准。

| 路线 | 状态 / 分支 | 模型 mean | 模型 p95 | 完整 policy mean | baseline / 模型 | 固定输入 cosine / RMSE |
|---|---|---:|---:|---:|---:|---:|
| OpenPI baseline | 可运行 / `main` | 918.45 | 945.19 | 927.91 | 1.00 | N/A / N/A |
| FlashRT BF16 | 可运行 / `flashrt` | 232.20 | 233.02 | 240.41 | 3.96 | 0.999789 / 0.008613 |
| FlashRT INT8（实验） | 误差过大，不采用 / `flashrt` | 167.67 | 168.22 | 174.11 | 5.48 | 0.580660 / 0.415863 |
| vLLM / vLLM-Omni | 当前栈未跑通 / `vllm` | N/A | N/A | N/A | N/A | N/A / N/A |
| TensorRT Edge-LLM | 当前栈未跑通 / `tensorrt-edge-llm` | N/A | N/A | N/A | N/A | N/A / N/A |

INT8 因固定输入动作偏差过大而不采用；未宣称通过或违反预先规定的任务精度门限。其延迟仅作实验记录，不作为可替代 baseline 的可用方案。余弦/RMSE 仅覆盖同一合成观测与固定噪声，不代表机器人任务成功率。

## 分步骤延迟

表内为 mean / p95。Vision 包括实际计算的 3 路图像，denoise 包括 10 次迭代。baseline 来自 CUDA event 插桩；FlashRT 来自独立诊断阶段图，不能将各阶段相加宣称等于未经插桩生产图。baseline denoise 区间不含随后 Euler 更新和循环控制。输入/输出边界为另行同步诊断数据。

| 路线 | 步骤 | mean / p95 (ms) |
|---|---|---:|
| OpenPI baseline | 输入准备+上传 | 6.92 / 7.18 |
| OpenPI baseline | 其中 CPU preprocess | 0.50 / 0.54 |
| OpenPI baseline | 其中 H2D+Observation | 6.41 / 6.67 |
| OpenPI baseline | Vision ×3 | 83.46 / 86.95 |
| OpenPI baseline | Prefill | 290.42 / 302.93 |
| OpenPI baseline | Denoise ×10 | 546.63 / 573.53 |
| OpenPI baseline | D2H+postprocess | 0.66 / 0.77 |
| FlashRT BF16 | 输入准备+上传 | 7.36 / 11.85 |
| FlashRT BF16 | 其中 CPU preprocess | N/A / N/A |
| FlashRT BF16 | 其中 H2D+Observation | N/A / N/A |
| FlashRT BF16 | Vision ×3 | 40.81 / 40.97 |
| FlashRT BF16 | Prefill | 140.96 / 141.47 |
| FlashRT BF16 | Denoise ×10 | 50.04 / 50.53 |
| FlashRT BF16 | D2H+postprocess | 1.16 / 1.95 |
| FlashRT INT8（实验） | 输入准备+上传 | 5.94 / 9.15 |
| FlashRT INT8（实验） | 其中 CPU preprocess | N/A / N/A |
| FlashRT INT8（实验） | 其中 H2D+Observation | N/A / N/A |
| FlashRT INT8（实验） | Vision ×3 | 35.00 / 35.09 |
| FlashRT INT8（实验） | Prefill | 96.10 / 96.26 |
| FlashRT INT8（实验） | Denoise ×10 | 35.68 / 35.93 |
| FlashRT INT8（实验） | D2H+postprocess | 1.14 / 1.47 |
| vLLM / vLLM-Omni | 全部步骤，未完成模型推理 | N/A |
| TensorRT Edge-LLM | 全部步骤，未完成模型推理 | N/A |

## 内存：生产稳态与完整模型 pass

各 backend 为独立新进程。稳态取 `steady_state_after_warmup`，发生在额外诊断图捕获之前。NvMap 为当前 Python PID 的 Jetson 驱动记账；RSS/PSS/USS 为进程主机映射。Jetson 统一内存的这些记账域存在重叠，不能相加。Torch 峰值只涵盖 Torch allocator，FlashRT 原生 cudaMalloc 与图/库缓冲区可能在其外。

| 路线 | 稳态 NvMap PID | RSS / PSS / USS | 稳态 Torch live / reserved | 模型 Torch allocator 峰值 | 模型 NvMap 观察最大值（下界） | 全设备 used（背景参考） |
|---|---:|---:|---:|---:|---:|---:|
| OpenPI baseline | 7,532.41 | 2,989.92 / 2,965.37 / 2,953.98 | 7,146.39 / 7,406.00 | 7,291.39 | 7,532.41 | 17,344.56 |
| FlashRT BF16 | 13,092.79 | 1,011.52 / 986.90 / 974.84 | 6,582.95 / 12,124.00 | 6,582.95 | 13,092.79 | 21,046.96 |
| FlashRT INT8（实验） | 13,059.91 | 1,068.65 / 1,043.97 / 1,031.87 | 8,779.99 / 12,126.00 | 8,779.99 | 13,059.91 | 21,108.19 |
| vLLM / vLLM-Omni | N/A | N/A / N/A / N/A | N/A / N/A | N/A | N/A | N/A |
| TensorRT Edge-LLM | N/A | N/A / N/A / N/A | N/A / N/A | N/A | N/A | N/A |

模型总峰值与阶段峰值来自不同 pass。总模型 pass 内不插入任何 stage peak reset；加载、图准备/捕获、model_total 的主机/NvMap 轮询等待间隔为 100 ms，读取本身另需约 40–52 ms，实际采样间隔更长；其他细分阶段只读取同步前后边界。这些观察最大值可能漏过瞬态尖峰，仅为峰值下界。全设备 used 包含其他进程和驱动；不代表当前模型占用，也不能用它的差值宣称精确进程峰值。内存 pass 含逐阶段同步、读取 /proc、sudo 与 profiler 本身开销，其时间不用于延迟结论。

## 各阶段内存

各行均是绝对占用或峰值高于该阶段起点的增量，单位 MiB。阶段共享权重、KV 和 scratch；预分配引擎阶段间增量接近零是正常现象，不能将行相加。backend_load / prepare_and_capture 包含加载与生产图准备；diagnostic_graph_capture 及其后的阶段含额外诊断图对象。

| 路线 | 阶段 | NvMap PID 观察最大绝对值 | RSS / PSS 观察最大值 | Torch allocator 最大绝对峰值 | Torch 峰值高于阶段起点 |
|---|---|---:|---:|---:|---:|
| OpenPI baseline | `backend_load` | 7,255.85 | 15,397.94 / 15,379.56 | 7,138.26 | 7,138.26 |
| OpenPI baseline | `cpu_preprocess` | 7,532.41 | 2,989.93 / 2,965.37 | 7,146.39 | 0.00 |
| OpenPI baseline | `d2h_and_postprocess` | 7,532.41 | 2,989.93 / 2,965.38 | 7,148.12 | 0.00 |
| OpenPI baseline | `denoise_iteration` | 7,532.41 | 2,989.93 / 2,965.38 | 7,182.26 | 4.34 |
| OpenPI baseline | `h2d_and_observation` | 7,532.41 | 2,989.93 / 2,965.37 | 7,148.83 | 2.45 |
| OpenPI baseline | `language_embedding` | 7,532.41 | 2,989.93 / 2,965.37 | 7,151.90 | 0.78 |
| OpenPI baseline | `model_preprocess` | 7,532.41 | 2,989.93 / 2,965.37 | 7,148.12 | 0.00 |
| OpenPI baseline | `model_total` | 7,532.41 | 2,989.92 / 2,965.37 | 7,291.39 | 143.28 |
| OpenPI baseline | `prefill` | 7,532.41 | 2,989.93 / 2,965.37 | 7,291.39 | 135.02 |
| OpenPI baseline | `vision` | 7,532.41 | 2,989.93 / 2,965.37 | 7,159.03 | 8.91 |
| FlashRT BF16 | `backend_load` | 12,489.91 | 7,114.63 / 7,095.62 | 12,057.85 | 12,057.85 |
| FlashRT BF16 | `d2h_and_postprocess` | 13,128.79 | 1,015.60 / 990.97 | 6,582.96 | 0.00 |
| FlashRT BF16 | `denoise` | 13,128.79 | 1,015.55 / 990.93 | 6,582.95 | 0.00 |
| FlashRT BF16 | `diagnostic_graph_capture` | 13,128.79 | 1,013.77 / 989.15 | 6,582.95 | 0.00 |
| FlashRT BF16 | `input_prepare_and_staging` | 13,128.79 | 1,015.50 / 990.87 | 6,582.96 | 0.00 |
| FlashRT BF16 | `model_total` | 13,092.79 | 1,011.68 / 987.06 | 6,582.95 | 0.00 |
| FlashRT BF16 | `prefill` | 13,128.79 | 1,015.55 / 990.92 | 6,582.95 | 0.00 |
| FlashRT BF16 | `prepare_and_capture` | 13,092.79 | 1,009.40 / 984.78 | 6,584.68 | 1.75 |
| FlashRT BF16 | `vision` | 13,128.79 | 1,015.52 / 990.89 | 6,582.95 | 0.00 |
| FlashRT INT8（实验） | `backend_load` | 12,491.98 | 7,032.45 / 7,013.45 | 12,057.85 | 12,057.85 |
| FlashRT INT8（实验） | `d2h_and_postprocess` | 13,093.91 | 1,075.15 / 1,050.47 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `denoise` | 13,093.91 | 1,075.14 / 1,050.46 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `diagnostic_graph_capture` | 13,093.91 | 1,070.27 / 1,045.59 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `input_prepare_and_staging` | 13,093.91 | 1,075.11 / 1,050.43 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `model_total` | 13,059.91 | 1,068.89 / 1,044.21 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `prefill` | 13,093.91 | 1,075.14 / 1,050.46 | 8,779.99 | 0.00 |
| FlashRT INT8（实验） | `prepare_and_capture` | 13,059.91 | 1,068.48 / 1,043.80 | 8,781.71 | 1.75 |
| FlashRT INT8（实验） | `vision` | 13,093.91 | 1,075.14 / 1,050.46 | 8,779.99 | 0.00 |
| vLLM / vLLM-Omni | 未获得阶段内存测量 | N/A | N/A | N/A | N/A |
| TensorRT Edge-LLM | 未获得阶段内存测量 | N/A | N/A | N/A | N/A |

## 状态与数据来源

- **FlashRT INT8（实验）**：同一合成输入下 cosine 0.580660、RMSE 0.415863，动作误差过大，本轮不采用；未设定或声称通过任务级精度门限。
- **vLLM / vLLM-Omni**：官方 Pi05Pipeline 存在，但当前 NVIDIA Torch 2.3 / Python 3.10 与新版本依赖不兼容；Orin 上导入及隔离 resolver dry-run 未进入模型构建。推理延时和模型内存均 N/A。
- **TensorRT Edge-LLM**：官方 v0.11.0 原生 π0.5 支持 LIBERO，但本机 TRT 8.6 缺少 >=10.15 所需 API，glibc 2.35 不满足 wheel 2.39；实际包解析和四项 C++ API 探针失败。未执行推理，延时/模型内存 N/A。
- OpenPI baseline latency：[summary.json](baseline-latency/summary.json)（`reports/engine-comparison-2026-10-16/baseline-latency/summary.json`；同目录保留 raw/provenance）。
- OpenPI baseline memory：[summary.json](baseline-memory/summary.json)（`reports/engine-comparison-2026-10-16/baseline-memory/summary.json`；同目录保留 raw/provenance）。
- FlashRT BF16 latency：[summary.json](flashrt-bf16-latency/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-bf16-latency/summary.json`；同目录保留 raw/provenance）。
- FlashRT BF16 memory：[summary.json](flashrt-bf16-memory/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-bf16-memory/summary.json`；同目录保留 raw/provenance）。
- FlashRT BF16 pipeline：[summary.json](flashrt-bf16-pipeline-latency/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-bf16-pipeline-latency/summary.json`；同目录保留 raw/provenance）。
- FlashRT INT8（实验） latency：[summary.json](flashrt-int8-latency/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-int8-latency/summary.json`；同目录保留 raw/provenance）。
- FlashRT INT8（实验） memory：[summary.json](flashrt-int8-memory/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-int8-memory/summary.json`；同目录保留 raw/provenance）。
- FlashRT INT8（实验） pipeline：[summary.json](flashrt-int8-pipeline-latency/summary.json)（`reports/engine-comparison-2026-10-16/flashrt-int8-pipeline-latency/summary.json`；同目录保留 raw/provenance）。

CSV 为长表，保留每个指标的数值、单位、`unmeasured`、相对源路径与说明。缺失列保持空值；Blocked 路线不读取不存在的测量目录。正式技术路径及失败原因详见仓库 `summary.md`。
