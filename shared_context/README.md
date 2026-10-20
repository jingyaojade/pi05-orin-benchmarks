# π0.5 + Qwen3.5-9B：同进程、共享CUDA上下文

新分支`vla-vlm-shared-context`，独立目录；对照实现`dual_stack/`保持原样。
继续使用原FlashRT BF16和同一GGUF Q5_K_M/FP16 mmproj，不改变模型几何和权重。

`bridge.cpp`提供C ABI，直接调用已验证版本的`libllama-server-impl.so`；
HTTP/SSE服务器代码仍是同一C++实现，但运行在Python进程的原生线程中，没有独立llama-server模型进程。
ctypes.CDLL在原生调用期间释放GIL。两个工作线程显式绑定Torch初始化的同一CUDA context。
VLA使用自身stream/event同步，下载转换也放在该stream；没有在每请求中等待整个设备。

## 构建、验证和测量

需要已有`dual_stack/`部署、FlashRT隔离venv、CUDA12.2及已编译的llama.cpp。
新增shim使用系统g++、CUDA/CUPTI头文件，不改系统软件或原llama动态库。
运行预检/benchmark前，如已启用本候选服务，先执行`bash shared_context/rollback.sh`；
测试恢复器管理原来的三个unit，不会自动停止新候选unit。

```bash
cd /home/nvidia/pi05-orin-benchmarks
bash shared_context/build_bridge.sh

# CUPTI callbacks在这台Orin上要求特权。只用于独立诊断，不用于正式延时。
sudo -n env HOME=/home/nvidia OMP_NUM_THREADS=6 \
  /home/nvidia/pi05-orin-flashrt/.venv/bin/python \
  shared_context/preflight.py --output /home/nvidia/pi05-orin-shared-context/new-preflight

# 新的空目录，普通用户测量。暂时停止本项目模型服务，退出时恢复。
/home/nvidia/pi05-orin-flashrt/.venv/bin/python -u shared_context/benchmark.py \
  --output reports/shared-context-new-run --memory-probe
python3 shared_context/report.py reports/shared-context-new-run
# 绘图在单独matplotlib环境执行即可。
python shared_context/plot_results.py reports/shared-context-new-run
```

默认每个变体：空闲30s、加载/预热、双常驻空闲30s、仅VLA60s、仅VLM60s、
持续并发至少180s、指定VLA2Hz/VLM0.2Hz约60s。
持续压测是每模型一个在途请求、完成后立即下一次；指定频率错过的触发直接跳过。
两种频率不是机器人关节控制率，也不是服务内部自动触发配置。
可另传`--variants shared_device`作为设备级同步诊断；主对照为`two_process shared_stream`。

独立CUPTI预检捕获实际VLA graph launch和Qwen launch的context，要求身份相同、角色均有记录。
它不是只比较初始化前后的context，也不是根据同一PID推断；诊断开销排除在主测之外。
共享上下文不保证所有kernel并行，CUDA计算与带宽仍共享，C++后端内部同步未全面重写。

## 功能范围

保持VLA `POST /infer`及VLM `/v1/chat/completions`，API图片/动作规格与双进程版相同。
正常未传noise时每次新采样；仅benchmark固定noise。图像逐次准备上传。
**本实验版本固定任务prompt。** Qwen常驻期间不执行VLA重捕获，变更prompt明确返回409；
本候选版尚未提供任务prompt配置入口或维护调度API，单纯重启仍使用原fixture任务。
这一限制避免活动的Qwen线程与VLA图重捕获冲突，不能作为支持任意任务切换的服务。
新增GET `/metadata`、`/stats`、`/audit/start`、`/audit/stop`；audit需特权，只用于诊断。

## 内存与监控

沿用相同1Hz tegrastats、thermal sysfs、proc、NvMap、MemAvailable、Swap与OC计数。
共享进程只记一次PID内存，不能将同一PID重复标为两个模型再求和。
Torch allocator只覆盖Torch分配，不包括所有llama.cpp CUDA分配；NvMap/PSS/RSS与系统RAM也有重叠。
功耗只加三条主轨，不重复加DDR VDDQ或SUMMATION，未测交流插座功耗。

`--memory-probe`增加单slot/context16384的常驻与短请求检查，和8K对比PID分配，
不代表完成16K输入长度评测。更多显存可用于更长context、更高精度或更高图像token/批量，
这些目的需要分别衡量质量、吞吐和延时，不能通过空分配提升推理速度。

## 候选服务切换与回退

完成报告、上下文证据收集且源码/二进制hash匹配后，可切换同一组回环端口：

```bash
bash shared_context/activate.sh reports/shared-context-2026-10-20
# VLA 8081、VLM 8082、兼容WebSocket 8000，均为同一Python PID。
bash shared_context/rollback.sh
# 完整切换、接口、固定prompt边界和回退测试（完成后仍为原双进程服务）：
/home/nvidia/pi05-orin-flashrt/.venv/bin/python shared_context/lifecycle_test.py \
  reports/shared-context-2026-10-20
```

切换保存原服务状态、保留原unit，新增`openpi-vla-vlm-shared`；回退按保存状态恢复。
固定prompt限制与性能结果应一起评估，测试脚本结束时恢复原双进程服务。
正式测评见[报告](../reports/shared-context-2026-10-20/report_zh.md)。

实测持续并发时：VLA模型427.71→428.22ms，VLM完整回答2856.64→2302.75ms；
VLA p95 473.68→504.69ms，指定2Hz时完成率1.807→1.640Hz。
因此当前保留该版本为已验证候选，默认恢复原双进程服务；没有承诺降低动作更新尾延时。
