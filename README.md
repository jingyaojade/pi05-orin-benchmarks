# π0.5 LIBERO on Jetson AGX Orin

记录已部署 OpenPI PyTorch 基线与 FlashRT 的分阶段推理耗时。

2026-10-20：已部署 **π0.5 FlashRT BF16 + Qwen3.5-9B Q5_K_M / FP16 vision** 双服务，
代码位于[dual_stack](dual_stack/README.md)，分支为 `vla-vlm-qwen35`。
VLA单独/同时推理的模型均值为 **220.80 / 427.42 ms**；VLM首字 **688.39 / 1350.02 ms**。
并发系统RAM峰值 **28.17 GiB**、GPU最高 **73.28°C**、板载三轨功耗平均 **45.82 W**。
[完整测试报告与时间轴](reports/dual-stack-2026-10-20-r2/report_zh.md)记录单独、双常驻、同时推理各场景及p95。
部署为systemd自启服务，保留8000 WebSocket，新增VLA HTTP 8081和VLM兼容API 8082，均监听回环。
原baseline服务已保存配置并停用，可通过`dual_stack/restore_original.sh`恢复。

2026-10-16一轮包含独立内存补测、vLLM/Omni 和 TensorRT Edge-LLM 实际兼容性探针：
[所有技术路径与结论](summary.md)、[分步骤延时与内存对比](reports/engine-comparison-2026-10-16/comparison.md)。
同轮纯模型 baseline **918.45 ms**，FlashRT BF16 **232.20 ms（3.96×）**；
稳态 PID NvMap 驱动记账分别为 **7.36 / 12.79 GiB**，不是独占物理内存总和。
vLLM/Omni 与 Edge-LLM 在当前软件栈未进入模型推理，性能为 N/A。
新引擎各自位于 `vllm`、`tensorrt-edge-llm` 分支；综合代码和本轮数据在 `engine-comparison`。
下文保留首轮历史数据，不能与新轮拼接计算加速比。

分支约定：`main` 保存 baseline 代码、原始测量和报告；`flashrt` 在 baseline 提交之上增加独立的 FlashRT 目录和对比结果。

本分支已完成 FlashRT BF16：纯模型 **230.918 ms/action chunk**，相对初次 / 相邻重测 baseline 分别加速 **3.807× / 4.207×**。动作余弦相似度 **0.999833**，RMSE **0.007630**；仅验证固定合成输入。
INT8 虽更快，但这组输入上的动作偏差较大，本次不选用。详见[完整对照结论](reports/flashrt-comparison_zh.md)与[FlashRT 复现步骤](flashrt/README.md)。

## Baseline 实测（2026-10-16）

Jetson AGX Orin 64GB、CUDA 12.2、NVIDIA PyTorch 2.3；每阶段预热 10 次、测量 30 次。

| 区间 | 均值 | p95 |
| --- | ---: | ---: |
| 完整本机策略（含预处理与数据传输，不含网络） | 886.895 ms | 891.282 ms |
| CUDA 输入就绪至 CUDA 动作输出完成的模型墙钟 | 879.124 ms | 882.884 ms |

每次输出 `10×7` action chunk，模型执行 10 次去噪。模型墙钟包含 Python 发射和循环开销。
分阶段计时、插桩增幅、硬件频率限制和数值检查见[完整中文报告](reports/baseline-2026-10-16/report_zh.md)。
[原始记录](reports/baseline-2026-10-16/raw.json)与[来源信息](reports/baseline-2026-10-16/provenance.json)一并保留。

在已有 Orin 部署上重测：`bash run_baseline.sh reports/baseline-new-run`。

## 测量口径

- 相同官方 `pi05_libero` checkpoint、固定观测和固定噪声。
- 预测 10 步动作、每步 7 维；模型内部执行 10 次去噪。
- 输入含两路有效相机与一路被掩码的填充图像，文本容量 200 tokens。
- 模型主指标为 CUDA 输入就绪至 CUDA 动作输出就绪，单独测量 CPU 预处理、传输和后处理。
- 整体性能与分阶段插桩分开采样；记录插桩开销，嵌套阶段不重复求和。
- 预热 10 次，默认每个测量阶段采样 30 次，保留每次测量数据。
- 基准期间暂时停止服务以避免 GPU 争用，结束后恢复原服务状态。

所有性能数字由真实 GPU 测量生成；运行脚本和生成报告程序分别位于 `baseline/` 与 `reporting/`。

## 依赖与来源

已有部署目录：`/home/nvidia/pi05-orin`，包括独立 Python 环境、上游源码和模型。
本仓库的 `deployment/` 保存策略加载和依赖配置；模型和第三方运行库不纳入 Git。

- OpenPI: https://github.com/Physical-Intelligence/openpi
- OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- FlashRT: https://github.com/flashrt-project/FlashRT

测量使用合成观测，只验证性能和数值一致性，不代表实体机器人任务成功率。
