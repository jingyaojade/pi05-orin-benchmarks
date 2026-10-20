# π0.5 + Qwen3.5-9B 双服务

分支 `vla-vlm-qwen35`。本目录保存本轮独立部署、API、测试与监控代码；
权重、第三方源码、构建产物位于 Orin `/home/nvidia/pi05-orin-dual`，不进入 Git。

VLA 为已验证补丁的 π0.5 LIBERO / FlashRT BF16，仍是三路视觉（两路有效）、
200 token 容量、10×32 模型动作、10轮去噪、最终10×7动作。VLM 为官方 Qwen3.5-9B
架构的公开 Unsloth Q5_K_M 语言权重及匹配 FP16 vision/mmproj；仅凭语言 GGUF 无法看图。
`sources.json` 固定 llama.cpp、GGUF revision、每个模型文件的精确字节数与 SHA256。
下载可通过 HF mirror 传输相同 revision，最终以原 HF 元数据 SHA256 校验。

## 隔离构建

```bash
cd /home/nvidia/pi05-orin-benchmarks
python3 dual_stack/download_models.py --output /home/nvidia/pi05-orin-dual/models/Qwen3.5-9B
bash dual_stack/setup_orin.sh
/home/nvidia/.local/bin/uv pip install \
  --python /home/nvidia/pi05-orin-flashrt/.venv/bin/python \
  -r dual_stack/requirements-orin-extra.txt
```

构建固定 SM87、现有 CUDA12.2；使用已有独立 FlashRT venv 的 CMake，
仅启用本次 KV 类型所需的 `GGML_CUDA_FA_QUANTS=f16-f16`，这不改变语言权重 Q5_K_M。
单GPU不启用 NCCL。不会给 baseline 安装新版 Torch，也不修改 JetPack/驱动/功耗/风扇。
默认8个编译任务，可用 `BUILD_JOBS` 调小。已完成的对象可复用。

## 工作负载与测试

`make_fixtures.py` 在本地生成两张512×384合成桌面图，没有文字标签；
是看图接口与性能 smoke test，不是机器人数据集。VLA复用已有固定合成相机/任务/noise。
两种模型无实体机器人连接，也没有让 VLM 自动生成并执行机器人指令。

```bash
python dual_stack/make_fixtures.py
/home/nvidia/pi05-orin-flashrt/.venv/bin/python dual_stack/run_suite.py \
  --output reports/dual-stack-new-run \
  --phase-seconds 60 --concurrent-seconds 180 --idle-seconds 30
python3 dual_stack/report.py reports/dual-stack-new-run
# Optional: matplotlib 3.9.4 in a separate plotting environment.
python dual_stack/plot_results.py reports/dual-stack-new-run
```

测试暂时停止本项目原 `openpi-pi05`，顺序为系统空闲、VLA单独运行、VLM单独运行、
双常驻空闲、双常驻仅VLA、双常驻仅VLM、双模型同时持续推理。
VLA启动包含10次warmup，client再排除5次；VLM排除2次warmup。
每个VLA phase至少30个请求，每个VLM phase至少8个请求；并发phase至少180秒。
请求通过本机HTTP，包含VLA两张224×224 PNG的解码；不是外部网络RTT。
两侧都预先准备PNG/base64，不计相机采集与客户端图像编码。
VLA计时含客户端JSON序列化；VLM从已构造HTTP请求开始，不含客户端JSON序列化。
原服务的起始active状态在退出时恢复，包括只针对start-limit-hit的频率计数恢复。
已存在的非空结果目录不得覆盖。

VLA每次请求重新准备/上传图像和固定noise；原完整生产图计时两端同步，
下载/反归一化单独计时。固定prompt持续缓存；改变prompt会显式重新prepare/capture，
该setup计入请求时间并标记，当前性能测试不改变prompt。
各VLA实例单独要求重复输出逐元素一致，并记录相对官方baseline的误差。

VLM context8192、单slot、4 CPU线程、FP16 KV、GPU offload、关闭thinking与prompt cache，
温度0、seed42、最多96生成tokens。两张图轮换，保留每个SSE数据包和实际输出长度。
TTFT以第一个非空content计时，不把role空包当首token。llama.cpp回传prompt/decode timings
单独保存；TTFT包含图像编码、prefill、排队、HTTP。视觉编码未被伪造为独立精确数值。

## 内存、热与功耗边界

`telemetry.py` 运行自己拥有的tegrastats子进程，1Hz读取当前电源轨、温度、GPU负载、
系统RAM，同时读取当前VLA/VLM PID的RSS/PSS/USS、NvMap、MemAvailable/Swap。
不使用 `tegrastats --stop` 杀其他人的监控。保存原始行、proc和NvMap表以及单调时钟。
另每秒直接读取cpu/gpu/tj/tboard thermal sysfs，保存原始毫摄氏度及不可读状态；
避免GPU休眠时启动tegrastats导致后续一直省略GPU温度。没有读数时保留缺失值。
逐秒保存OC硬件事件计数；计数不增长本身不足以排除所有热/电限频。
读取本身有开销且非原子采样，观测峰值是瞬态峰值下界；UMA指标存在重叠，不能相加。

AGX Orin的板载三轨合计为 VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0。
VDDQ_VDD2_1V8AO 已包含在 VIN_SYS_5V0，SUMMATION也与前两轨重叠，均不能再加。
这是板载电源轨近似值，包含背景系统负载，不是交流插座/整机适配器精密功率计读数。
固定顺序、动态频率、短时持续负载不能替代随机交叉试验与长期热稳态测试。

依据：[NVIDIA功耗传感器说明](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling)、
[tegrastats](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)、
[llama.cpp多模态](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)。

## 持久服务与恢复

仅在GPU测试成功后执行：

```bash
bash dual_stack/install_services.sh reports/dual-stack-2026-10-20-r2
```

该步骤保存原服务active/enabled状态，保留原unit与源码；停止并取消原baseline服务自启，
避免同一π0.5权重重复加载。新增并启用 `openpi-pi05-flashrt-dual`（HTTP 127.0.0.1:8081）
和 `qwen35-vlm`（OpenAI兼容HTTP 127.0.0.1:8082）。VLA还保留原OpenPI msgpack WebSocket
`127.0.0.1:8000`及`/healthz`，由同一进程转发到HTTP VLA端口，不额外加载模型。
该兼容入口要求224×224 RGB，并增加PNG编码/loopback转发开销，正式组合延时测的是8081 HTTP入口。
已有无关同名unit不会被覆盖，安装失败会尝试恢复原状态。

VLA：GET `/health`、`/metadata`、`/stats`；POST `/infer`，JSON字段为
`image_base64`、`wrist_image_base64`（PNG/JPEG解码后须224×224 RGB）、`prompt`，
可选 `state[8]`、`noise[10,32]`；未指定noise时每请求重新生成FP32标准正态噪声。
返回actions[10,7]及组件timings。测试显式传固定noise，正常服务不固定采样噪声。
VLM：GET `/health`、`/metrics`；POST `/v1/chat/completions`，支持image_url data URI和SSE。
端口只监听回环，可从VS Code端口转发访问。

恢复原WebSocket服务：

```bash
bash dual_stack/restore_original.sh
```

该命令停止/取消新增双服务自启，按保存状态恢复原服务，不删除模型与测试数据。

CPU检查：`python -m unittest discover -s dual_stack/tests -v`；这些检查只验证记账/计时处理逻辑，
实际部署成功与性能以Orin原始结果为准。

WebSocket兼容入口另在FlashRT隔离venv安装`websockets==15.0.1`、`msgpack==1.1.2`，
复用已部署的官方openpi_client/msgpack_numpy.py helper；不会导入baseline的OpenPI/JAX模型。

实际完成的测量见[中文报告](../reports/dual-stack-2026-10-20-r2/report_zh.md)。
首轮因发现温度采样遗漏而主动终止，未当作完成的报告；Orin保留原始未完成目录。
正式结果为增加直接温度采样后重新执行的完整r2。
