# Orin π0.5 选择性量化与补充候选实验

本目录是独立实验工具，不修改现有推理服务、旧适配器或固定 FlashRT vendor 源码。
量化以 BF16 数据流为基准，只替换指定 GEMM，保留原 GELU、Norm、残差和时间积分。
旧原生 INT8 路径中的 SiLU EVT 不参与本实验。

## 环境与输入

- Jetson AGX Orin 64GB，现有 CUDA 12.2 / NVIDIA PyTorch 独立环境。
- 固定 FlashRT `6cb2f80caec2f561a1331c554a6d3bbf95d2ed55` 及仓库 `flashrt/patches/manifest.json`。
- 使用既有 `pi05_libero` 检查点、同一 tokenizer、10 步 action chunk、32 维内部动作，返回 7 维。
- 固定 17 个数值样本：历史合成输入 1 个；FlashRT 附带 8 组 LIBERO 图像，每组 2 份显式固定噪声。
- 图像统一使用基准 prompt `pick up the black bowl`；不是按任务描述运行的机器人 episode。
- 三路有效输入实验复制 front 图像作为第三路，只验证形状、性能和数值；不代表真实三相机任务质量。

## 运行

在包含本目录、`baseline/` 和 `flashrt/` 的代码目录运行：

```bash
ROOT=/home/nvidia/pi05-orin-selective-quantization-20261030
PYTHON=/home/nvidia/pi05-orin-flashrt/.venv/bin/python
$PYTHON selective_quantization/prepare_inputs.py \
  --asset /home/nvidia/pi05-orin-flashrt/vendor/FlashRT/examples/thor/assets/pi05_libero.npz \
  --legacy reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz \
  --output "$ROOT/inputs"
python3 selective_quantization/run_suite.py --root "$ROOT" &
SUITE_PID=$!
python3 selective_quantization/retry_measurements.py --root "$ROOT" \
  --parent-pid "$SUITE_PID" \
  --extra-configs selective_quantization/additional_configs.json
wait "$SUITE_PID"
$PYTHON selective_quantization/report.py --results "$ROOT/results" --output "$ROOT/report"
```

`run_suite.py` 会暂时停止其记录为 active/activating 的既有模型服务，在 finally 中恢复原服务状态。
不修改时钟设置或服务配置；同一 GPU 上各实验串行运行。

## 实验设计

1. 先使用原 OpenPI 环境生成两路有效与三路有效的 10 步参考动作。
2. 同一进程预量化所有候选权重，分别只启用 encoder/decoder 的一个 block，筛选共 36 层。
3. 单层筛选使用历史样本及前四帧的首份噪声；组合结果另行检查所有 17 个样本，并单列留出样本。
4. 筛选规则 `min per-case cosine >= 0.999`、`RMSE <= 0.02`、`max abs <= 0.1` 只是工程数值筛选规则，不是任务成功门限。
5. 最终组合各用新进程，仅准备选中权重，记录 10 次预热、50 次主计时、独立 30 秒持续负载遥测。
6. 补充候选：968→775 仅去文本 padding；968→519 同时移除 masked 第三相机；三路有效独立比较968→775；去噪10→8/5步仍保留10步动作chunk。
7. 独立阶段图在主计时及内存遥测之后建立，并检查其输出与原完整图完全一致。
8. 结束时再运行 BF16 控制组，显示顺序、温度和运行时 GEMM tactic 漂移的影响。
9. 追加探索组：decoder FFN + compact519；decoder FFN + 三路有效 compact775；保守 encoder 15/16/17 层独立测试及其组合。保守集合使用 discovery 每个样本 cosine>=0.9999、RMSE<=0.005、max_abs<=0.025；该规则是在初始组合误差出现后提出的探索规则，不是独立确认性结论。

`retry_measurements.py` 会等主进程结束后再执行，不会同时运行 GPU 实验。早期遥测无效或失败的配置写入独立 `_retry` 目录，不覆盖初始 BF16 teacher 动作或旧日志。

## 指标边界

- 主延迟：GPU 输入准备完成→原完整 CUDA Graph→GPU 同步完成，不含上传、下载、反归一化与网络。
- 另记录 Vision / Prefill / Denoise 诊断图；其和不替代原完整图墙钟。
- 内存：RSS、NvMap、Torch allocated/reserved、CUDA device 分开保存，不相加。选择性代理保留 BF16 原权重并增加 INT8 副本，不是内存压缩版部署。
- 温度：报告持续负载窗口 CPU/GPU 起始、结束、最大温度；30 秒不是热稳态，不能把顺序差异当作降温效果。
- cosine/RMSE/max-abs 是相同输入及噪声下的动作数值一致性；本实验没有机器人/仿真闭环成功率结论。
- 原始样本、计时、遥测、量化权重选择及失败日志均保留，失败配置不从报告中静默删除。
