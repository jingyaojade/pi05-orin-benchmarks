# 基线性能中文报告生成器

`report.py` 使用 Python 标准库读取 `baseline/benchmark.py` 的产物，不执行推理、不连接设备、不修改测量数据。

```bash
python reporting/report.py \
  --results results/baseline \
  --output reports/baseline_zh.md
```

`--results` 也可以写作 `--baseline`，接受结果目录或 `summary.json` 文件。未指定 `--output` 时，写入结果目录的 `report_zh.md`。再次运行会更新该 Markdown 文件。

可选对照使用相同的统计 schema：

```bash
python reporting/report.py \
  --results results/baseline \
  --compare results/flashrt \
  --output reports/comparison_zh.md
```

报告内容包括：

- 未插桩完整策略、模型同步墙钟与 CUDA 流事件区间的均值、中位数、p95、范围、标准差和样本数。
- 分阶段墙钟计时，以及已插桩与未插桩结果的差值和耗时比。
- 模型顶层互不重叠区间的占比，及前缀、去噪、后缀和时间 MLP 的嵌套层级。
- 原始记录中的阶段调用次数、逐相机与逐去噪迭代分布、动作形状和数值校验。
- 环境、版本、仓库提交/状态、权重与实际载入源码的 SHA-256。
- 可选对照的匹配指标和来源字段核对；不会自动宣称跨后端数值等价。

同时识别仓库中的 FlashRT harness：将其 `flashrt_policy_uninstrumented.policy_wall_ms` 明确映射到完整策略墙钟。独立生产模型计时直接读取 `model_uninstrumented.model_wall_ms` 与 `model_cuda_stream_elapsed_ms`，比较 CUDA 输入就绪到输出完成的相同边界；对应字段不存在时显示“未记录”。对照表列出每个指标的数据来源。

报告列出 FlashRT **每个 phase 的全部已测指标**，包括视觉编码、前缀 KV 预填充、全部去噪迭代、阶段链 CUDA 区间和诊断调用墙钟，各有样本数、均值、中位数、p95、范围与标准差。诊断阶段链存在时，会分别计算其流区间相对生产模型流区间、完整诊断调用墙钟相对生产完整策略墙钟的均值差与比例；不会混淆模型与含前后处理的调用边界。

报告展示 `accuracy.json` 中独立测得的动作误差，并引用实现记录的精度限制。单独捕获、分别发射的 `flashrt_diagnostic_stage_graphs` 具有不同的 launch 边界，不映射为生产模型耗时，也不会用于推断模型加速比。诊断链额外耗时也可能受捕获边界与温度/频率漂移影响，不等同于纯 launch 开销。同步模型墙钟仍含 replay 发射开销，CUDA Event 也不是 GPU 内核忙碌时间。

来源核对明确映射 FlashRT 权重、归一化和序列化 tokenizer 模型的散列。其 `hashes.fixture` 是读取的 **NPZ 文件散列**，只与本地基线 `fixed_inputs.npz` 文件散列比较，绝不与 `fixture.array_sha256` 混比。另逐数组比较不含文件头的内容散列，因此重存 ZIP 元数据而内容不变的 NPZ 仍能核对内容。两侧记录实际 token IDs/mask 时会逐项比较；一侧未记录时保留缺失。

如果原始基线 provenance 没有 token IDs，生成器可读取 `BASELINE_DIRECTORY.parent / baseline-tokenizer-audit.json` 的测后 CPU 审计 sidecar。仅在其 `model_sha256` 等于基线 `checkpoint.tokenizer_model_sha256`、`fixture_sha256` 等于本地基线 `fixed_inputs.npz` 的实际文件 SHA-256 时采用。基线已记录 tokenizer 源码散列时，还会验证 `source_sha256`；同时检查文本指令、完整 200 个 IDs/mask 及有效 IDs 的内部一致性。报告明确标为“补充 CPU 审计，非 benchmark 当时记录”，并保留原始计时及 provenance 不变。绑定失败会显示原因，并拒绝使用该 sidecar 的 IDs/mask。

必需输入为 `summary.json`。同目录的 `raw.json` 和 `provenance.json` 用于详细调用计数、层级、数值校验与来源。`fixed_inputs.npz` 和 `reference_actions.npy` 存在时，生成器只读取安全的数组文件头来展示形状与 dtype，不载入 pickle 或对象数据。

缺少字段显示“未记录”。数值统计中的 NaN/Infinity、不合法的样本数及明显不符的 schema 会报错。单独提供 summary 时，原始计数、按父区间筛选的子阶段耗时和来源保持缺失，不会据此补零或估算。

CUDA Event 是流上的经过时间，可能包括 CPU 发射空隙，不是 GPU 内核忙碌时间。嵌套阶段不可相加。顶层占比以已插桩模型 CUDA 流区间为分母；它不能代替未插桩完整策略基线。报告也会明确说明，十步动作不是十次模型请求，顺序请求吞吐不是机器人的控制频率。
