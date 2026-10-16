# vLLM / vLLM-Omni：Orin π0.5 兼容性实测

当前 Orin 软件栈不适合把 vLLM 作为现有 LIBERO 策略的直接加速后端。已执行源码、导入和依赖解析探针；没有进入模型构建或 GPU 推理，因此本路径的延时、峰值内存和加速比均为 **N/A**，不是 0。原始证据在 [compatibility.json](results/2026-10-16/compatibility.json)。

## 三种路径必须区分

| 路径 | π0.5 action expert / flow matching | 本机结论 |
|---|---|---|
| 普通 vLLM 的 PaliGemma | 普通 VLM 不包含当前 AdaRMS expert、10 步 Euler 或连续动作头 | 不能用文本生成吞吐代替策略测量 |
| 官方 vLLM-Omni `Pi05Pipeline` | 有完整 π0.5 数学实现与机器人服务接口 | 当前依赖无法与保留的 NVIDIA Torch 2.3 共存 |
| 单独拷出 Omni 数学模块后改为普通 Torch 执行 | 可以另做源码移植 | 不是已运行的 vLLM 服务或原生 CUDA 加速，未计为成功引擎 |

截至 2026-10-16，官方 Omni 的 π0.5 recipe 已存在，不能笼统声称“vLLM 不支持 π0.5”。但该 recipe 明确定位于功能实现，使用 eager，尚无此模型的 CUDA Graph、Triton、跨请求 prefix/encoder cache。其已验证硬件列出 H100、RTX 4090 和 RTX PRO 5000；这不构成 Orin 性能证明。见[锁定的官方 recipe](https://github.com/vllm-project/vllm-omni/blob/90576488ae85de6ac355bb7aa3e9470c38303bba/recipes/lerobot/Pi05.md)。

## 已执行的尝试

时间：2026-10-16 15:07:31（机器实际 UTC 时间以 JSON `timestamp` 为准）。所有操作均在 `/home/nvidia/pi05-orin-vllm`，未安装新包进 baseline 环境，也没有改驱动、JetPack、CUDA 或电源配置。

| 项目 | 机器实际结果 |
|---|---|
| Python / 架构 | 3.10.12 / aarch64 |
| L4T / CUDA toolkit | R36.3 / 12.2.140 |
| NVIDIA Torch / Transformers | `2.3.0a0+6ddf5cf85e.nv24.4` / `4.53.2` |
| 官方 Pi05 原文件导入 | `ModuleNotFoundError: No module named 'vllm'`，发生在权重 loader 依赖导入，尚未构建模型 |
| 隔离空 venv，`uv pip install --dry-run`，vLLM / Omni 0.30.0 | resolver 发现 vLLM 要求 Torch 2.13.0，与保留 NVIDIA Torch 2.3 的 constraint 冲突 |
| 隔离空 venv，vLLM 0.31.0 release dry run | 同样被 Torch 2.13.0 与当前 constraint 冲突阻止 |
| 当前 vLLM HEAD 源码的 Python 元数据 | 要求 `>=3.11,<3.15`；本机 3.10.12 不满足 |
| 推理 / 逐步骤延时 / 推理峰值内存 | 未执行 / N/A / N/A |

源码 HEAD 与发布 wheel 必须区分：发布版 0.30.0 / 0.31.0 resolver 输出要求 Torch 2.13.0，而本次锁定的 vLLM HEAD 构建元数据已要求 Torch 2.14.0。没有把源码 HEAD 的要求伪称为 wheel 实测结果。CUDA 12.9 预编译包也不能因为存在 aarch64 构建就当作当前 Jetson CUDA 12.2 的兼容二进制。官方安装说明要求在 CUDA / Torch 不同时从源构建，见[官方安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。

`sources.json` 固定并校验每个审计文件的 SHA-256，包含：

- vLLM：`fa75d3456d60a5ffe923818c4b495295675b38e1`。
- vLLM-Omni：`90576488ae85de6ac355bb7aa3e9470c38303bba`。
- vLLM `v0.5.0`：`8f89d72090da70895d77d32248ea8504f7daba50`。该旧版明确使用 Torch 2.3.0，但其模型 registry 没有 Pi05，不能通过降级包获得今天的完整 Pi05 后端。见[旧版依赖](https://github.com/vllm-project/vllm/blob/8f89d72090da70895d77d32248ea8504f7daba50/requirements-cuda.txt)与[旧版 registry](https://github.com/vllm-project/vllm/blob/8f89d72090da70895d77d32248ea8504f7daba50/vllm/model_executor/models/__init__.py)。这只证明所检查的 Torch 2.3 对应 tag，未声称穷举所有下游分支。

## 为什么即使换栈也不能直接用官方 recipe 做当前比较

当前测试对象是 OpenPI `pi05_libero`，action horizon 10、模型 action dim 32、对外输出 10×7，两个真实相机加一个 masked 相机 slot，文本固定容量 200，且 `discrete_state_input=False`。官方 Omni 的现成 recipe 针对 `lerobot/pi05_base`，输出 50×32，并在 prompt 中序列化 state。其配置没有当前 LIBERO 的 `discrete_state_input` 开关。直接替换权重并发送 state 会改变 token 内容，不能比较为同一任务。

若后续在新栈移植，需完成以下技术路径：

1. 保持本次 `fixed_inputs.npz`、固定 FP32 噪声、三相机顺序及 mask；复用 baseline 的确切 tokenizer、量化归一化和输出反归一化。
2. 使用官方 `Pi05ForActionPrediction.sample_actions(images, image_masks, lang_tokens, lang_masks, noise, num_steps=10)` 的底层输入边界，或给 `Pi05Processor` 增加并验证“不序列化 state”的 LIBERO 分支；不能沿用 50×32 base 策略输入。
3. 将 OpenPI 权重严格映射到官方 Omni 模块，要求零 missing / skipped 权重，校验权重 hash。无需随机初始化任何 expert 参数。
4. 对齐每个参数的 BF16 / FP32 dtype。官方 Omni 的 mixed precision 函数将整个 vision tower 和 projector 保留 FP32，与当前 OpenPI 的选定 FP32 参数规则不同，需要分别记录再决定同精度比较。
5. 验证三路视觉、968 prefix rows、bidirectional prefix、prefix KV 与 AdaRMS 条件，核对固定输入动作的 RMSE / cosine / max error。
6. 在真正的 `Pi05Pipeline` / DiffusionEngine 完整路径运行，而不是对 `modeling_pi05.py` 的 Torch 拷贝贴上 vLLM 标签；如单独测试数学模块，应独立命名。
7. 按共同规范重测 GPU-ready→GPU-ready 模型时间、独立视觉 / prefix / 10 步 denoise、完整 policy 时间，以及 CPU RSS、Torch allocated / reserved 和 Jetson 系统 RAM。前缀视觉随每帧变化，不能把跨观测的缓存复用当作无条件加速。

上述移植与新栈推理尚未执行，不提供预计延时或推理内存数字。对当前设备的结论是继续采用已有 FlashRT 对照结果，vLLM 路径保留为明确可重放的兼容性尝试。

## 复现

```bash
cd /home/nvidia/pi05-orin-benchmarks
bash vllm/run_probe.sh /home/nvidia/pi05-orin-vllm/new-results
```

输出目录必须不存在。`probe.py` 只下载公开锁定源码、校验 hash、执行原文件 import 和依赖 dry-run；会创建独立 `.venv-probe`，不会往其中安装包，也不显式调用 CUDA API、构建模型或分配 CUDA 张量。探针的墙钟运行时间不属于推理性能。

`serve_official_recipe.sh` 提供新兼容环境中的官方 **base** 策略启动命令，未在 Orin 执行。它要求显式配置本地标准 LeRobot 模型目录与匹配的 Omni 源码，不下载模型、不自动迁移 host。由于它输出 50×32，不能用于本次 LIBERO 10×7 对照表。
