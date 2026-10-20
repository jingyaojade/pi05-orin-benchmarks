# π0.5 + Qwen：Linux NVIDIA RTX Docker构建包

目标为 **x86_64 Linux + RTX 4090 / RTX 5090**，分支 `docker-nvidia`。
**当前交付源码构建包：本地没有Docker和目标GPU，尚未构建镜像、运行模型或测得RTX性能。**
原Thor构建包保留在 `docker-thor` 分支和原归档中；本包入口是本文件。
无需连接原Orin来准备构建代码；从Orin提取已有权重时才需要连接它。

## 环境与后端

| 项目 | 本包配置 |
|---|---|
| 宿主机 | x86_64 Linux，Docker + NVIDIA Container Toolkit，支持实际显卡的NVIDIA驱动 |
| 基础镜像 | `nvcr.io/nvidia/pytorch:25.03-py3`，构建时解析digest并留档 |
| 用户态 | CUDA12.8.1、Python3.12、NGC PyTorch2.7.0a0；保留基础Torch |
| llama.cpp编译目标 | 默认 `89;120`，覆盖RTX4090 SM89与RTX5090 SM120 |
| VLA | 官方π0.5 LIBERO，OpenPI PyTorch BF16，10轮去噪 |
| VLM | Qwen3.5-9B Q5_K_M语言权重 + FP16 vision，llama.cpp CUDA |
| 进程 | 独立Python VLA进程 + 原生llama-server；支持单卡共用或双卡各一个模型 |

GPU架构见[NVIDIA计算能力表](https://developer.nvidia.com/cuda/gpus)。
CUDA/Python/Torch版本及R570或更新驱动要求见[NGC 25.03发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)。
驱动还须支持目标显卡的具体型号；不把数据中心卡的旧驱动兼容例外套到GeForce。
宿主准备参照[NVIDIA Container Toolkit安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
Docker不包含宿主机GPU驱动，也不会自动安装驱动、修改Docker daemon或调整功耗/风扇。

不是所有NVIDIA GPU的通用镜像。其他架构要调整编译目标，并重新完成GPU与模型验收。
FlashRT源码/补丁随包提供，但原适配器仍有SM87限制；本包没有移植或验收RTX FlashRT，
没有复制Orin/Thor的CUDA二进制。RTX PyTorch与Orin FlashRT的比值不能作为纯硬件加速比。

## 1. 模型文件

五个固定资产沿用Thor包及原Orin测试SHA，约14.73GB，可直接复用已有导出目录。
权重独立只读挂载，不放Git、源码归档或Docker镜像，不在构建时下载。
在已有模型文件的机器运行：

```bash
python3 containers/assets.py export \
  --pi05 /absolute/path/pi05_libero \
  --qwen /absolute/path/Qwen3.5-9B \
  --tokenizer /absolute/path/paligemma_tokenizer.model \
  --output /absolute/path/pi05-qwen-assets
python3 containers/assets.py verify /absolute/path/pi05-qwen-assets
```

输出必须是新目录，先校验来源再复制、复验，保留 `asset-manifest.json`。
也可从其他已有副本按 `pins.json` 的目录与文件SHA组织资产，无需从Orin导出。

## 2. 构建

在目标x86_64 Linux解压本包，或切换Git的 `docker-nvidia` 分支，在仓库根目录运行：

```bash
bash containers/nvidia/build.sh
# 可选：只编译4090或5090，减少编译量；不设置则默认同时覆盖两者。
# CUDA_ARCHITECTURES=89 bash containers/nvidia/build.sh
# CUDA_ARCHITECTURES=120 bash containers/nvidia/build.sh
# IMAGE_TAG=pi05-qwen:rtx BUILD_JOBS=2 bash containers/nvidia/build.sh
```

需要访问NGC、GitHub及Python依赖源；没有离线依赖下载器或跨CPU仿真。
默认镜像名 `pi05-qwen:nvidia-candidate`；若自定义IMAGE_TAG，后续run也须设同值。
OpenPI/llama.cpp固定commit，安装固定4.53.2的官方transformers_replace，llama.cpp从源码构建。
独立venv继承NGC Torch，并检查其版本未被依赖安装替换；不安装Orin的Torch wheel。
构建信息保存在 `containers/local-artifacts/nvidia-build-*`；镜像内保留：

- `/opt/requirements-built.txt`：本次完整pip freeze。
- `/opt/benchmark-source-sha256.json`：源码SHA。
- 镜像revision label、固定上游commit，以及构建脚本记录的基础digest和镜像inspect。

原OpenPI依赖中仍有范围约束，首次真实构建后才可固化完整transitive lock。
当前不能承诺不同日期重建得到相同镜像字节。构建本身不要求加载模型或连接原Orin。

## 3. 单卡验收与测量

```bash
# 无需权重：检查架构/所选GPU，逐卡执行真实BF16矩阵乘法。
bash containers/nvidia/run.sh doctor

export ASSETS_DIR=/absolute/path/pi05-qwen-assets
export RESULTS_DIR=/absolute/path/rtx-results
bash containers/nvidia/run.sh verify-assets
bash containers/nvidia/run.sh smoke
bash containers/nvidia/run.sh baseline
bash containers/nvidia/run.sh benchmark
```

默认 `VLA_GPU=0 VLM_GPU=0`：两个独立进程共用一张卡。
doctor的UUID与设备名保存在测量环境中，模型进程分别记录实际可见设备。
CUDA探针在短命子进程中执行并退出，再启动模型；监督进程不会为探针持续持有CUDA上下文。

`smoke`检查HTTP/SSE/WebSocket、图像变化和动态任务输入，不代表实体机器人任务精度。
VLA记录与原reference actions的误差，同一实例的固定输入要求逐元素一致；跨GPU不要求bitwise一致。
24/32GB等容量是否足够双模型及加载峰值，须以目标机smoke与遥测为准。

`baseline`是单VLA的原分层插桩，默认每阶段预热10、测量30次，
保存CPU预处理/传输、模型、vision/prefix/10轮去噪、后处理等原始记录。
`benchmark`测双常驻空闲、仅VLA活动、仅VLM活动、持续并发及周期触发：
默认单模型场景60s（至少VLA30/VLM8次）、并发180s、空闲30s、周期60s。
周期目标为VLA2Hz、VLM0.2Hz；错过时隙跳过，报告实际完成率。
VLA两路有效224×224相机+一路masked、文本容量200、固定noise、10×7动作；
VLM沿用两张512×384图片、8K上下文单slot、最大96输出token、thinking/cache关闭。

每次创建UTC时间/命令/GPU编号结果目录，不覆盖已有数据。
组合测量保存各请求API/组件延时、Qwen TTFT及prefill/decode时间、mean/p95、吞吐、原始SSE和服务日志。
VLA组合服务只拆分预处理/上传/整体模型/下载；内部层耗时须另运行baseline，不能把两轮样本拼成同次请求。
完整测试成功后生成 `report_zh.md`、`summary.json`；失败保留 `completed:false`、原因与原始数据。
测试前请清空其他GPU负载；脚本不自动停止宿主或其他容器的服务。

## 4. 双卡各运行一个模型

```bash
VLA_GPU=0 VLM_GPU=1 bash containers/nvidia/run.sh doctor
VLA_GPU=0 VLM_GPU=1 bash containers/nvidia/run.sh smoke
VLA_GPU=0 VLM_GPU=1 bash containers/nvidia/run.sh benchmark
```

参数为容器可见的CUDA逻辑序号，doctor会检查所选两卡并记录UUID。
`run.sh`默认暴露所有GPU，不继承宿主机的CUDA_VISIBLE_DEVICES过滤；请以doctor输出为准。
运行时为每个模型单独设置CUDA_VISIBLE_DEVICES，因此各子进程内部只见自己的 `cuda:0`。
如果绕过run.sh直接docker run并设置父CUDA_VISIBLE_DEVICES，选择序号会相对这个集合重映射，
包括UUID列表；语义参照[CUDA环境变量说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#env-vars)。
双卡拆分不做模型跨卡切分、不要求NVLink；性能仍需实测，不能保证独占卡就完全无干扰。

单卡/双卡应保持相同权重、fixture和阶段参数，并同时保留型号、UUID、驱动、时钟和温度。
混用4090/5090时还要区分模型分配方案，不能把差异全归因于去掉同卡争用。

## 5. 内存、温度与功耗

每秒采集 `telemetry.jsonl`，GPU指标按UUID分别汇总，避免双卡读数混合：

- 每张设备：NVML/nvidia-smi memory.used/total（MiB）、温度（°C）、设备功耗（W）、GPU活跃率和时钟。
- 每个模型进程：RSS/PSS及原始proc文件；VLA另记Torch live/reserved/allocator peak。
- 主机：MemAvailable和可读取的hwmon原始值；不盲目对传感器求和。

设备级读数可能含其他任务，主机内存不是进程专属；RSS/PSS、Torch allocator、VRAM不能直接相加。
独显VRAM与Orin统一内存的记账范围不同，不能用一个百分比直接比较。
GPU设备功耗不等于整机AC功耗，也不是Orin板载三轨合计；测整机需外置功率计。
不支持的读数保留N/A及查询错误，1Hz采样可能遗漏瞬态峰值。
固定测试顺序与动态时钟不构成热稳态认证；不得用未测量的零值填报告。

## 6. 服务与再分发

```bash
bash containers/nvidia/run.sh serve
# 双卡服务：VLA_GPU=0 VLM_GPU=1 bash containers/nvidia/run.sh serve
```

宿主机只映射127.0.0.1:8081（VLA HTTP）、8082（Qwen API）、8000（OpenPI WebSocket）。
服务按请求触发，默认没有定时循环。退出只清理自己的子进程。
不用privileged、宿主PID命名空间或systemd挂载，不改变现有Orin服务。
serve与benchmark不要同时占用同一GPU测量。

生成源码包（在Git仓库中、代码已提交后）：

```bash
python3 containers/package_buildkit.py --target nvidia --output /path/pi05-qwen-nvidia-buildkit.tar.gz
sha256sum -c /path/pi05-qwen-nvidia-buildkit.tar.gz.sha256 # 在归档所在目录校验
tar -xzf /path/pi05-qwen-nvidia-buildkit.tar.gz
cd pi05-qwen-nvidia-buildkit
```

归档来自Git提交，带文件SHA和manifest，不带权重、.git、账号或编译产物。
历史测量报告不全量打包，查看Git仓库对应分支；本包没有RTX实测报告。
目标GPU完成构建、doctor、接口/数值及性能验收后，可导出真正运行镜像：

```bash
docker save pi05-qwen:nvidia-candidate -o pi05-qwen-nvidia-image.tar
# 其他兼容Linux/NVIDIA机器：docker load -i pi05-qwen-nvidia-image.tar
# 单独复制权重目录，在新机再次执行doctor、verify-assets、smoke后测量。
```

CPU准备检查见[validation_2026-10-20.md](validation_2026-10-20.md)。
下一阶段待目标机到位：实际构建、完整依赖锁、单/双卡模型与监控验收、实测报告、镜像归档；
RTX FlashRT支持属于独立移植任务，目前没有宣称已完成。
