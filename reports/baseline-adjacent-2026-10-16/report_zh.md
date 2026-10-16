# π0.5 Orin 基线性能报告

测量记录时间：2026-10-16 13:21:24 UTC+08:00。数据目录：`D:\openpi-main\pi05-orin-benchmarks\reports\baseline-adjacent-2026-10-16`。

未插桩完整策略平均耗时 **968.795 ms/请求**；CUDA 输入就绪到输出计算完成的模型平均墙钟耗时 **971.478 ms**。
由完整策略均值换算的顺序请求吞吐约 **1.032 次/秒**，不是机器人的控制频率。

这些结果使用固定合成输入与显式固定噪声；未运行机器人任务成功率评估。完整策略基线不含网络通信。

## 测试设置与数值校验

| 项目 | 记录 |
| --- | --- |
| 种子 | 20261016 |
| 每阶段配置的测量次数 / 排除的预热次数 | 30 / 10 |
| 批大小 / 动作预测步数 / 去噪迭代次数 | 1 / 10 / 10 |
| 编码图像数 / 有效图像数 | 3 / 2 |
| torch.compile 配置 | 关闭（None） |
| 文本指令 | pick up the black bowl |

10 步动作与 10 次去噪是两个独立设置。占位相机虽被注意力掩码屏蔽，仍执行视觉编码；本基线保留该行为。

| 阶段 | 实际测量样本数 | 记录的预热样本数（排除） |
| --- | --- | --- |
| 官方完整策略（未插桩） | 30 | 10 |
| 模型（无内部插桩） | 30 | 10 |
| 分解测量（已插桩） | 30 | 10 |

原始记录中的动作形状计数：`{'[10, 7]': 90}`；有限值校验：`{'true': 90}`；与参考动作逐位相等校验：`{'true': 90}`。

汇总中的全部动作逐位相等标记：`True`。包装恢复后验证：`{'shape': [10, 7], 'finite': True, 'bitwise_equal_reference': True, 'max_abs_error': 0.0}`。

保存的数组文件头（只读取元数据，不反序列化对象）：

| 数组 | 形状 | dtype |
| --- | --- | --- |
| image | `(224, 224, 3)` | `\|u1` |
| wrist_image | `(224, 224, 3)` | `\|u1` |
| state | `(8,)` | `<f4` |
| noise | `(10, 32)` | `<f4` |
| prompt | `()` | `<U22` |
| reference_actions | `(10, 7)` | `<f8` |

### 基线 tokenizer 补充 CPU 审计

审计文件：`D:\openpi-main\pi05-orin-benchmarks\reports\baseline-tokenizer-audit.json`；审计记录时间：2026-10-16 12:56:01 UTC+08:00。

这是性能测量完成后的 CPU tokenization 审计，**不是 benchmark 当时记录**，不补写或更改原始计时/provenance 文件。

声明范围：post-measurement CPU-only tokenization audit; baseline timing files unchanged。接受状态：**通过核对，接受为补充证据**。

| 绑定检查 | 基线值 SHA-256 前缀 | sidecar 值 SHA-256 前缀 | 核对 |
| --- | --- | --- | --- |
| tokenizer 模型 SHA-256 | `8986bb4f423f07f8…` | `8986bb4f423f07f8…` | 一致 |
| 基线实际 NPZ 文件 SHA-256 | `c7b18b3b0dbbccdc…` | `c7b18b3b0dbbccdc…` | 一致 |
| 官方 tokenizer 源码 SHA-256 | `965be8b3c393a681…` | `965be8b3c393a681…` | 一致 |

补充审计中的有效 token 数：7；有效 IDs：`[2, 18075, 908, 573, 2656, 14581, 108]`。

## 未插桩基线分布

| 计时区间 | n | 均值 ms | 中位数 ms | p95 ms | 最小 ms | 最大 ms | 标准差 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 完整策略墙钟 | 30 | 968.795 | 970.025 | 984.903 | 951.089 | 998.947 | 11.770 |
| 模型同步墙钟 | 30 | 971.478 | 971.691 | 1006.392 | 944.066 | 1016.238 | 18.431 |
| 模型 CUDA 流事件区间 | 30 | 971.433 | 971.642 | 1006.343 | 944.023 | 1016.199 | 18.430 |

完整策略计时包含 CPU 输入处理、输入传输、模型采样、输出传回与动作解码。模型墙钟计时前输入已在 CUDA 上就绪，计时结束前同步整个设备；因此其中也包含 Python 发射与循环控制开销。

## 插桩开销与完整分解

| 比较 | 未插桩均值 ms | 已插桩均值 ms | 差值 ms | 耗时比 | 均值增幅 |
| --- | --- | --- | --- | --- | --- |
| 模型墙钟 | 971.478 | 982.528 | 11.050 | 1.0114× | 1.14% |
| 完整策略 vs 同步分解总墙钟 | 968.795 | 990.234 | 21.440 | 1.0221× | 2.21% |

完整分解额外在输入传输结束及完整模型结束处同步。它与未插桩策略的差异同时包含事件包装、边界同步及分阶段运行时的温度/频率漂移，不能全部解释为包装函数的成本。

| 分解区间 | n | 均值 ms | 中位数 ms | p95 ms | 最小 ms | 最大 ms | 标准差 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU 预处理 | 30 | 0.512 | 0.502 | 0.596 | 0.447 | 0.661 | 0.043 |
| 输入 H2D 与 Observation 构建 | 30 | 6.516 | 6.492 | 7.190 | 6.112 | 7.549 | 0.314 |
| 模型同步墙钟 | 30 | 982.528 | 976.692 | 1006.503 | 951.957 | 1114.973 | 27.744 |
| 输出 D2H 与后处理 | 30 | 0.678 | 0.666 | 0.834 | 0.578 | 0.880 | 0.069 |
| 分解总墙钟 | 30 | 990.234 | 984.139 | 1014.263 | 959.510 | 1122.820 | 27.844 |

## 模型顶层互不重叠区间

以下百分比以已插桩模型的 CUDA 流事件区间均值为分母，只选择互不重叠的顶层区间。残余区间来自实测总区间减去这四组顶层区间。

| 互不重叠区间 | 每请求区间均值 ms | 占模型流区间 |
| --- | --- | --- |
| 模型内部预处理 | 0.181 | 0.02% |
| 完整前缀嵌入 | 91.128 | 9.28% |
| VLM 前缀 KV 预填充 | 297.109 | 30.24% |
| 动作去噪迭代 | 587.948 | 59.85% |
| 剩余区间（掩码/位置、Euler 更新、循环与空隙等） | 6.077 | 0.62% |
| 模型总 CUDA 流事件区间 | 982.441 | 100.00% |

CUDA Event 测量当前流上两次事件之间的经过时间，其中可能包含等待 CPU 发射造成的空隙。它不是 GPU 内核忙碌时间或利用率；模型总区间也不应被称为纯 GPU 运算时间。

## 嵌套区间层级

子区间已包含在父区间中，跨层级不可相加。表中父区间占比使用每请求累计区间均值，而不是单次调用均值。

| 父区间 | 子区间 | 每请求累计均值 ms | 占父区间 |
| --- | --- | --- | --- |
| 完整前缀嵌入 | 视觉编码（全部视角） | 90.100 | 98.87% |
| 完整前缀嵌入 | 语言 token 查表 | 0.153 | 0.17% |
| 完整前缀嵌入 | 该父区间其余部分 | 0.874 | 0.96% |
| 动作去噪迭代 | 动作与时间后缀嵌入 | 17.214 | 2.93% |
| 动作去噪迭代 | 注意力掩码数值转换 | 1.957 | 0.33% |
| 动作去噪迭代 | 动作专家 Transformer | 557.113 | 94.76% |
| 动作去噪迭代 | 动作输出投影 | 1.373 | 0.23% |
| 动作去噪迭代 | 该父区间其余部分 | 10.291 | 1.75% |
| 动作与时间后缀嵌入 | 动作输入投影 | 1.453 | 8.44% |
| 动作与时间后缀嵌入 | 完整时间 MLP（含两次 SiLU） | 4.223 | 24.53% |
| 动作与时间后缀嵌入 | 该父区间其余部分 | 11.538 | 67.03% |
| 完整时间 MLP（含两次 SiLU） | 时间 MLP 输入线性层 | 1.019 | 24.12% |
| 完整时间 MLP（含两次 SiLU） | 时间 MLP 输出线性层 | 0.966 | 22.87% |
| 完整时间 MLP（含两次 SiLU） | 该父区间其余部分 | 2.238 | 53.00% |

时间 MLP 总区间包含两次 SiLU，单独线性层区间不包含激活。其余部分还可能含 Python 发射与事件空隙，不能直接当作某个未单独测量算子的耗时。去噪迭代区间不含后续 Euler 更新及循环条件检查。

## 阶段调用次数与分布

调用次数来自已插桩原始记录。单次调用分布与每请求累计区间分开列出；例如视觉每请求执行三次，去噪每请求执行十次。

| 区间 | 调用/请求 | 总调用数 | 单次均值 ms | 单次中位数 ms | 单次 p95 ms | 每请求累计均值 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 模型内部预处理 | 1 | 30 | 0.181 | 0.179 | 0.208 | 0.181 |
| 完整前缀嵌入 | 1 | 30 | 91.128 | 89.408 | 103.622 | 91.128 |
| 视觉编码（全部视角） | 3 | 90 | 30.033 | 29.412 | 36.098 | 90.100 |
| 语言 token 查表 | 1 | 30 | 0.153 | 0.164 | 0.176 | 0.153 |
| VLM 前缀 KV 预填充 | 1 | 30 | 297.109 | 296.158 | 306.792 | 297.109 |
| 动作去噪迭代 | 10 | 300 | 58.795 | 57.584 | 66.309 | 587.948 |
| 动作与时间后缀嵌入 | 10 | 300 | 1.721 | 1.669 | 2.166 | 17.214 |
| 注意力掩码数值转换 | 11 | 330 | 0.197 | 0.193 | 0.224 | 2.164 |
| 动作专家 Transformer | 10 | 300 | 55.711 | 54.593 | 62.243 | 557.113 |
| 动作输入投影 | 10 | 300 | 0.145 | 0.134 | 0.232 | 1.453 |
| 动作输出投影 | 10 | 300 | 0.137 | 0.136 | 0.154 | 1.373 |
| 完整时间 MLP（含两次 SiLU） | 10 | 300 | 0.422 | 0.413 | 0.475 | 4.223 |
| 时间 MLP 输入线性层 | 10 | 300 | 0.102 | 0.099 | 0.123 | 1.019 |
| 时间 MLP 输出线性层 | 10 | 300 | 0.097 | 0.094 | 0.111 | 0.966 |

### 逐相机视觉编码

| 区间 | n | 均值 ms | 中位数 ms | p95 ms | 最小 ms | 最大 ms | 标准差 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 主相机 | 30 | 31.697 | 30.920 | 36.281 | 30.308 | 39.268 | 2.103 |
| 左腕相机 | 30 | 30.029 | 28.939 | 36.098 | 27.180 | 39.379 | 2.789 |
| 右腕占位图像（mask=false） | 30 | 28.373 | 28.577 | 29.599 | 22.637 | 37.345 | 2.354 |

### 逐次去噪迭代

| 区间 | n | 均值 ms | 中位数 ms | p95 ms | 最小 ms | 最大 ms | 标准差 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 次 | 30 | 65.803 | 63.928 | 78.869 | 57.644 | 89.010 | 7.308 |
| 第 2 次 | 30 | 58.737 | 57.597 | 61.582 | 55.650 | 82.861 | 4.713 |
| 第 3 次 | 30 | 58.379 | 57.225 | 59.421 | 55.463 | 88.219 | 5.635 |
| 第 4 次 | 30 | 58.361 | 57.396 | 59.624 | 55.200 | 82.944 | 4.683 |
| 第 5 次 | 30 | 58.002 | 57.433 | 59.861 | 55.921 | 72.121 | 2.824 |
| 第 6 次 | 30 | 57.480 | 57.220 | 59.278 | 55.561 | 62.315 | 1.338 |
| 第 7 次 | 30 | 57.929 | 57.630 | 60.235 | 56.001 | 61.759 | 1.245 |
| 第 8 次 | 30 | 57.758 | 57.645 | 59.430 | 56.071 | 60.147 | 1.057 |
| 第 9 次 | 30 | 57.648 | 57.282 | 60.044 | 55.879 | 61.586 | 1.372 |
| 第 10 次 | 30 | 57.851 | 57.384 | 60.539 | 55.728 | 62.715 | 1.534 |

## 环境与来源

| 项目 | 记录 |
| --- | --- |
| 设备 | Orin |
| 系统 | Linux-5.15.136-tegra-aarch64-with-glibc2.35 |
| 计算能力 | `[8, 7]` |
| 统一/设备内存总量 GiB | 61.363 |
| 峰值 CUDA 张量分配 GiB | 7.121 |
| CUDA / cuDNN | 12.2 / 8904 |
| Python | 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] |
| PyTorch / Transformers | 2.3.0a0+6ddf5cf85e.nv24.4 / 4.53.2 |
| JAX / Flax / NumPy | 0.5.3 / 0.10.2 / 1.26.4 |
| 参数元素 dtype 分布 | `{'torch.float32': 119720608, 'torch.bfloat16': 3497036912}` |
| PyTorch 执行设置 | `{'num_threads': 6, 'num_interop_threads': 12, 'float32_matmul_precision': 'high', 'cudnn_benchmark': False, 'cudnn_deterministic': False, 'deterministic_algorithms': False, 'matmul_allow_tf32': True, 'cudnn_allow_tf32': True}` |

峰值 CUDA 张量分配不是整机统一内存使用量。功耗模式、频率、只读命令结果与环境变量见 provenance.json；本报告不假设运行时锁频。

| 仓库 | 提交 | 远程 | 工作区状态 |
| --- | --- | --- | --- |
| openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` | https://github.com/Physical-Intelligence/openpi.git | `M src/openpi/shared/download.py` |
| benchmark | 未记录 | 未记录 | 未记录 |

权重目录：`/home/nvidia/pi05-orin/models/pi05_libero`。

| 资产 | SHA-256 |
| --- | --- |
| model.safetensors | `1502e1c5f765ca1bfd6f8f6be0bc769ba2cc263bf298b98bd106d8db168c02e6` |
| norm_stats.json | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| tokenizer model | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

实际载入源码：

| 源码 | 路径 | SHA-256 |
| --- | --- | --- |
| benchmark | `/home/nvidia/pi05-orin-benchmarks/baseline/benchmark.py` | `303e9281235c5fe8097628db033751ca0f785ec41e3764ef49afdca8f085f25b` |
| deployment_helper | `/home/nvidia/pi05-orin/serve.py` | `67d15284872d5c70b95e162b4102c8dcd6de4a8f79187e623c1f9e96291c6353` |
| policy | `/home/nvidia/pi05-orin/openpi/src/openpi/policies/policy.py` | `a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20` |
| model | `/home/nvidia/pi05-orin/openpi/src/openpi/models_pytorch/pi0_pytorch.py` | `dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16` |
| gemma | `/home/nvidia/pi05-orin/openpi/src/openpi/models_pytorch/gemma_pytorch.py` | `225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb` |
| patched_gemma | `/home/nvidia/pi05-orin/.venv/lib/python3.10/site-packages/transformers/models/gemma/modeling_gemma.py` | `17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde` |
| patched_siglip | `/home/nvidia/pi05-orin/.venv/lib/python3.10/site-packages/transformers/models/siglip/modeling_siglip.py` | `ef2e99500f263fdd78db9d4d9a255e29d844eb92f53c25ad640f8056700ef84b` |
| patched_paligemma | `/home/nvidia/pi05-orin/.venv/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py` | `c945d330b829264f927f3ee5339f977238631772de07a28e855e265bb85fd53c` |
| openpi.transforms | `/home/nvidia/pi05-orin/openpi/src/openpi/transforms.py` | `a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4` |
| openpi.models.pi0_config | `/home/nvidia/pi05-orin/openpi/src/openpi/models/pi0_config.py` | `d321858c30b398126035ea2b6da09e1c422eeeebb5ebf8df631daf9dc77647fa` |
| openpi.models.tokenizer | `/home/nvidia/pi05-orin/openpi/src/openpi/models/tokenizer.py` | `965be8b3c393a6811875bbc32da9e01a5d01cc2f87802de801cf7293e049748c` |
| openpi.models_pytorch.preprocessing_pytorch | `/home/nvidia/pi05-orin/openpi/src/openpi/models_pytorch/preprocessing_pytorch.py` | `6a44e91be2d7a724c6140d2cc98ed0a631b8d38bbb29f63b380a0a19f903bbf6` |
| openpi.policies.libero_policy | `/home/nvidia/pi05-orin/openpi/src/openpi/policies/libero_policy.py` | `d8679d99facc3fef48da07406081b20709ed872fcefdf7de67ede99422d623fa` |
| openpi.shared.normalize | `/home/nvidia/pi05-orin/openpi/src/openpi/shared/normalize.py` | `6c2cea4946fb07e51801530400d2b2fd94730e195cd290d6b6960114eca9739d` |

## 解读范围

- 主结论使用未插桩基线；插桩区间用于定位耗时，不能替代未插桩总耗时。
- 本机策略耗时排除 WebSocket、SSH、相机采集与机器人执行耗时。
- 固定合成输入可以检验数值一致性与时间分布，不能证明真实场景任务成功率。
- 各阶段顺序执行，温度、功耗与频率变化可能影响差异；不同阶段不是同时测得。
- p95 是有限样本的经验分位数；缺少的字段保持“未记录”，不会补成零或推断值。

原始依据：`summary.json`、`raw.json`、`provenance.json`、`samples.csv`、`stages.csv`、`fixed_inputs.npz` 与 `reference_actions.npy`。
