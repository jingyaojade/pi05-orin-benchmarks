# TensorRT Edge-LLM：π0.5 原生路径与当前 Orin 兼容性

2026-10-16 的结论：**模型结构适合，当前设备软件栈不能直接运行官方原生路径。** NVIDIA v0.11.0 已有实验性 π0.5 四组件实现，支持 `pi05_libero`；本次在 Orin 上实际执行了包解析和 C++ API 兼容性探针，尚未加载模型引擎。其推理延时、分阶段延时和模型内存均记为 **N/A**，不能用兼容性探针的运行时间/RSS 代替，也不能拿普通 TensorRT 的结果标成 Edge-LLM。

分支 `tensorrt-edge-llm`；所有本引擎代码、原始结果和说明在本目录。父目录已有的 baseline/FlashRT 文件来自共同基点。

## 固定上游版本与官方证据

- 仓库：[NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)。固定 release **v0.11.0**，完整 SHA **`95515c2f87fba8982db5a519f9022277667b3cc9`**。
- [固定版本 π0.5 示例](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/examples/vla/pi05.md)及[设计](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/developer_guide/models/pi05.md)：`visual`、`prefix`、`action`、`cond` 四组件，FP16，支持 LIBERO/DROID/ALOHA 的 openpi contract。
- [支持矩阵](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/getting_started/support-matrix.md)：该 release 的官方 Orin 目标为 JetPack 7.2 / CUDA 13.2 / 平台 TensorRT 10；aarch64 wheel 的 glibc 最低版本为 2.39，原生 payload 按平台、CUDA/TRT SONAME 与 SM 精确选择。
- [ONNX 原生算子 schema](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/tensorrt_edgellm/onnx/onnx_custom_schemas.py)：明确 `Attention` / `RotaryEmbedding` 需要 **TensorRT ≥ 10.15**。
- [action 插件](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/cpp/plugins/attentionPlugin/attentionPlugin.h)继承 `IPluginV3OneBuildV2`；[pi05 runtime](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/experimental_models/pi05/cpp/common/pi05Common.cpp)调用 `getDeviceMemorySizeV2()` / `setDeviceMemoryV2()`。
- `source_audit.json` 保存上述源码内容 SHA256、完整 pin 和历史 tag 树检查。v0.6.0（`996623c07270355dbf532f0600fe2626fce0704e`）与 v0.10.1（`e8b29522938901f6df19ebeedd4b69bc8edbcd97`）都没有 `experimental_models/pi05`；不能退回旧 JetPack 兼容版本后假定仍有原生 π0.5。

## Orin 上实际执行的尝试

原始结果：[compatibility.json](results/compatibility-2026-10-16/compatibility.json)，失败原因原文：[package_resolver.log](results/compatibility-2026-10-16/package_resolver.log)。

| 检查项 | 本机实测 | 原生路径要求/结果 |
|---|---|---|
| CPU 架构 | aarch64 | 匹配 |
| glibc | 2.35 | wheel 要求 2.39，解析器拒绝 |
| L4T | 36.3.0，Ubuntu 22.04 | 官方新路径目标 JetPack 7.2 |
| CUDA Toolkit | 12.2.140 | 官方 Orin payload 为 CUDA 13 |
| TensorRT | 8.6.2（Debian 8.6.2.3） | 原生 pi05 算子要求 ≥10.15 |
| `#include <NvInfer.h>` control | 编译成功 | 编译器、头文件路径有效 |
| `INetworkDefinition::addAttention` | 编译失败：成员不存在 | 原生 attention API 缺失 |
| `INetworkDefinition::addRotaryEmbedding` | 编译失败：成员不存在 | 原生 RoPE API 缺失 |
| `IPluginV3OneBuildV2` | 编译失败：类型不存在 | action 插件 ABI 缺失 |
| `ICudaEngine::getDeviceMemorySizeV2` | 编译失败：成员不存在 | runtime API 缺失 |
| `uv pip install --dry-run --no-deps tensorrt-edgellm==0.11.0` | 无匹配 `manylinux_2_35_aarch64` wheel | 真实执行，无包被安装 |
| 模型推理/模型内存 | 未执行 / N/A | 引擎不可加载，不能给性能数字 |

探针只读取系统信息、编译 C++ 声明并在独立空 venv 内做 dry-run；不初始化 CUDA、不加载权重、不安装包、不改系统 TensorRT/JetPack/驱动，也不改 baseline 环境。第一次 `python -m venv` 遇到宿主缺少 ensurepip，已改为已有的 `uv venv`；最终结果对应可复现的 uv 路径。

```bash
/usr/bin/python3 tensorrt_edge_llm/probe.py \
  --output /home/nvidia/pi05-orin-edge-llm/probe-2026-10-16 \
  --fetch-source --resolve-package
```

`--fetch-source` 只下载固定 SHA 的七个证据文件。若目标机访问 GitHub raw 较慢，可先在可联网主机取得这些文件并保留目录结构，再用 `--source-root /path/to/evidence` 代替；本次采用这一方式，文件 SHA 已记录。结果中的 `inference_executed:false` 与 `latency_ms:null` 是有意保留的状态。

## 隔离安装能解决什么

新 venv 不会改变 glibc、CUDA 驱动或 TensorRT C++ ABI。容器可带新 glibc/用户态库，但仍共享宿主 Jetson 驱动；仅启动新镜像不能证明当前 BSP 能加载官方 CUDA 13 payload。当前设备 Docker 可用，因此阻塞并非 Docker 未安装。

也不能把“CUDA 12.2”单独当作所有 TensorRT 10 的禁止条件：[TensorRT 10.15.1 通用发布说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.15.1.html)包含 CUDA 12.x 支持；[CUDA for Tegra 12.6](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html)也允许部分 JetPack 6.x CUDA 升级路径。这些事实**不等于**当前官方 Edge-LLM 的 Jetson Orin / SM87 / glibc / TRT payload 可无改动运行在 L4T 36.3 上。SBSA 的 aarch64 库也不能仅因 CPU 架构相同就当作 Jetson iGPU 发行版。

本次证据证明“当前已安装栈与官方发行路径不兼容”，未证明一切源码移植在理论上不可能。将 v0.11.0 回移到 CUDA 12/TRT 10、重做目标内核和包、验证 Jetson iGPU 及动作数值，会成为独立移植工程。将 π0.5 子图降解为 TRT 8.6 支持的 ONNX 基础算子，则属于自定义普通 TensorRT 后端，也不能标为 Edge-LLM 原生。为维持 baseline 和现有服务可用，本次未执行宿主升级或库替换。

## 为支持新栈准备的完整技术路径

`run_native_supported.sh` 是**未在本机执行的准备脚本**，带当前栈会失败的显式 guard。脚本固定源码 SHA，使用官方 Orin CMake toolchain、`EMBEDDED_TARGET=jetson-orin`、`CUDA_CTK_VERSION=13.2`、`ENABLE_CUTE_DSL=ALL`、`BUILD_EXPERIMENTAL_MODELS=ON`，构建 `pi05_policy_build` / `pi05_policy_inference`。

导出在新的独立环境完成；`export_requirements.txt` 来自固定 release 的 pyproject export extra（包括 Torch 2.13.0、Transformers 5.14.1），不是给现有 NVIDIA Torch 2.3 baseline venv 安装的依赖。可以在兼容 x86 CPU 导出主机上导出，再将同一 ONNX bundle 传给新 SDK 的 Orin 构建；脚本合并展示了完整步骤，未自动安装或升级系统 SDK。

权重始终是已部署的 `/home/nvidia/pi05-orin/models/pi05_libero/{model.safetensors,config.json}`，**不另下 LeRobot 权重**。原 baseline 权重 SHA256 是 `1502e1c5f765ca1bfd6f8f6be0bc769ba2cc263bf298b98bd106d8db168c02e6`。转换后的 config 声明 `action_dim=32`、`action_horizon=10`、`pi05=true`、`discrete_state_input=false`、`max_token_len=200`。脚本显式选择 `--pi05-policy-config pi05_libero`，从原部署复制同一份 quantile norm stats 到导出 bundle 的 `assets/norm_stats.json`，并保存权重、配置、fixture、norm stats 的 SHA256。

固定上游的 [weights.py](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/tensorrt_edgellm/models/pi05/weights.py)明确接收 openpi converter 的命名和 `action_dim/action_horizon` schema。当前没有运行原生 exporter，所以**尚未验证完整组件权重加载的 zero-missing/zero-unexpected，也没有验证每个源 tensor 到导出 tensor 的哈希映射**。新栈实跑时必须保留完整映射和 dtype 转换记录、检查所有必需 tensor 已加载（仅可排除官方明确不用的 expert lm_head、prefix final norm 等 dead weights）；缺失、shape 冲突或不一致应直接失败，不能随机初始化补齐或换下载另一个 checkpoint。本目录不声称这些待办已完成。

运行顺序：

1. 固定 checkpoint + FP16 导出 `visual/`、`prefix/`、`action/`、`cond/`，同时产生 `policy.json`、tokenizer、embedding sidecar。
2. 设置 `EDGELLM_PLUGIN_PATH` 指向同次构建的 `libNvInfer_edgellm_plugin.so`；原生 `action` 引擎解析和加载都需要它。
3. 构建全部四个组件；统一 export identity / policy contract，避免混用不同权重或 horizon。
4. `prepare_inputs.py` 将原 baseline `fixed_inputs.npz` 变为 FP16 两路 `[2,3,224,224]` pixels、相同任务 token IDs、**原始 FP32** `[1,10,32]` 初始 noise，以及原始观测 JSON。不能只设同一随机 seed，因为不同实现 RNG 不同。
5. 分别运行 canonical tensor 模式与 raw policy 模式，10 轮 denoise、10 次 warmup、30 次测量，保存动作输出。
6. `compare_actions.py` 对 native normalized `[1,10,32]` 使用原 quantile stats 还原前 7 个机器人动作维度，再与 baseline 的 `[10,7]` `reference_actions.npy` 比较 cosine/RMSE/MAE/max error。不能把 normalized 32 维与 robot-unit 7 维直接比较。默认只报告误差，不宣称机器人成功率，也不凭任意阈值判定可用。

新平台上的使用示例：

```bash
export EDGELLM_EXPORT_PYTHON=/path/to/new/export-venv/bin/python
export INPUT_PYTHON=/path/to/python-with-numpy-pillow-sentencepiece-safetensors
export FIXTURE=/home/nvidia/pi05-orin-benchmarks/reports/baseline-2026-10-16/fixed_inputs.npz
export CHECKPOINT=/home/nvidia/pi05-orin/models/pi05_libero
bash tensorrt_edge_llm/run_native_supported.sh
```

## 分阶段测量与公平性边界

官方 CLI 已提供 `visual`、`assemble`、`prefix`、`action (N steps)` stream-event 延时，以及 host 计时的 `engine chain`、观测前端和 `policy end-to-end`。`cond` 按 denoise 设置缓存；固定 10 步和 batch=1 时在启动/设置变化时计算，稳态 per-request 不再重算。未来测量必须同时记录其 cold/setup 成本。

Native runtime 会跳过被 mask 的第三相机和 padding tokens，prefix 长度为 `512 + 有效任务 token 数`；当前 baseline/匹配布局 FlashRT 做了三次 vision 和 968 行 dense prefix。compaction 在该双向 prefix + cumsum RoPE 语义下等价，但工作量减少，比较结果须将其作为优化说明，不能写成同 FLOPs 的纯 kernel 替换。Native 为 FP16，当前 baseline/FlashRT 基准为 BF16，数值需单独验证。

本地 CPU 输入适配验证已通过，结果在 [validation.json](results/input-adapter-validation-2026-10-16/validation.json)：原 fixture 的任务为 7 个有效 token，native compact prefix 会是 519 行。初始 noise 字节完全一致，pixels 布局/FP16 字节正确，token IDs 与实际 openpi `PaligemmaTokenizer.tokenize()` 方法一致。动作表示转换分别通过 robot JSON 恒等检查与 normalized safetensors 的 quantile 往返检查；这些检查使用明确标记的合成输出，只验证适配代码，不是 Edge-LLM 推理或其精度结果。

将来可比的纯模型指标应从 GPU 输入准备好开始，到完整 chunk GPU-ready 同步结束。官方 `engine chain` 的 host timer 包含 enqueue 间隙，与 stage CUDA-event 求和含义不同；canonical `policyMs` 也不能未经边界核对就和纯模型比值混算。各阶段中嵌套的时间不可重复相加。

内存必须记录整个推理进程的 RSS/VmHWM、可用时 CUDA allocator/device 用量，以及 Jetson 统一内存的系统 RAM 与空闲基线；普通 TensorRT/C++ 分配不经过 Torch allocator，`torch.cuda.max_memory_allocated()` 不能代表其全部模型内存。当前本引擎没有模型实例，因此本次各阶段和峰值模型内存均无有效数据。
