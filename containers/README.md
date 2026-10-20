# π0.5 + Qwen：Thor Docker构建包

**状态：已准备构建代码和CPU检查；尚无Thor，未构建镜像，未完成GPU验收，也没有Thor性能数字。**
本地没有Docker CLI/daemon；当前Orin SSH超时。这里交付的是源码构建包，不是已验证的`docker save`镜像。
解压后从本文件开始操作；历史大规模测量报告不全量放入构建包，可在Git仓库查看。

目标是Jetson Thor：两个独立模型进程、相同π0.5 LIBERO checkpoint、Qwen3.5-9B Q5_K_M + FP16 vision。
默认VLA明确使用OpenPI PyTorch BF16，10轮去噪，不静默回退或冒充FlashRT。
现有FlashRT SM87适配器、补丁与历史代码随包保留，但尚未移植到Thor。
本次没有修改原`dual_stack/`、`flashrt/`、baseline模型代码，也没有改变Orin正在运行的服务。

## 为什么需要Thor版本

| 项目 | 原Orin实测环境 | 本构建包的Thor候选 |
|---|---|---|
| CPU架构 | aarch64 | aarch64 |
| GPU计算能力 | SM87 / 8.7 | SM110 / 11.0 |
| 记录的软件栈 | L4T36.3、Ubuntu22.04、CUDA12.2 | NVIDIA PyTorch25.08容器：Ubuntu24.04、CUDA13.0 |
| Torch | NVIDIA2.3 nv24.04 | 基础容器预装2.8.0a0，保留不替换 |
| Qwen后端 | llama.cpp已编译SM87 | 同一commit重新编译SM110 |
| VLA后端 | FlashRT BF16 / 原PyTorch baseline | 先验证PyTorch BF16；FlashRT移植另验收 |

[GPU计算能力](https://developer.nvidia.com/cuda/gpus)、
[Thor CUDA官方说明](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/0.1.0/setup_cuda.html)、
[PyTorch25.08内容](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html)、
[Jetson框架兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)。
25.08是与JetPack7.0对应的固定候选基础版本，不声称适合所有未来Thor系统。
拿到设备后先记录JetPack/L4T与驱动，按官方兼容表决定是否覆盖`BASE_IMAGE`。
不能只复制Orin的venv、`.so`或将架构检查从87改为110。

## 包的组成

- Docker镜像：候选Thor用户态依赖、固定上游源码、按SM110编译的llama.cpp、测试代码。
- 独立权重目录：约14.73GB（13.72GiB）及少量tokenizer/norm stats，运行时只读挂载。
- 可写结果目录：服务日志、每次请求/SSE、环境、1Hz遥测、统计与报告。
- 所有固定版本/模型SHA见`pins.json`；沿用原实验fixture与noise。
- `orin-reference.json`保存已完成实验的Orin版本、FlashRT venv freeze和checkpoint记录，
  明确标为历史参考；不是本次实时SSH采集，也不是Thor依赖锁。

Docker不携带宿主机内核/GPU驱动。目标需要可用的Docker和NVIDIA Container Toolkit。
准备这个构建包不必连接Orin；从现有Orin提取权重或校准原生环境时需要访问它。
最终GPU验收和Thor性能测试必须在Thor上执行。

## 1. 准备权重

在拥有原始模型文件的机器运行，默认路径对应现有Orin部署，也可覆盖三个输入路径：

```bash
python3 containers/assets.py export --output /path/to/pi05-qwen-assets
# 自定义输入：--pi05 /path/pi05_libero --qwen /path/Qwen3.5-9B --tokenizer /path/paligemma_tokenizer.model
python3 containers/assets.py verify /path/to/pi05-qwen-assets
```

只复制五个指定文件，先核对原实验SHA，再复制、复验并生成`asset-manifest.json`。
输出必须是新目录，不覆盖旧权重；不复制账号、SSH、Git凭据或用户home。
支持用已有下载文件直接组织为`pins.json`所示目录，不要求一定从Orin导出。
将整个目录传给Thor即可。权重不进入Git、不在镜像构建时下载。

## 2. 构建镜像

在Thor/aarch64环境解压构建包或切换到`docker-thor`分支：

```bash
bash containers/build.sh
# 可选：BASE_IMAGE=官方兼容镜像 IMAGE_TAG=pi05-qwen:thor-my-system BUILD_JOBS=2 bash containers/build.sh
```

脚本拉取基础镜像并解析为digest，记录基础/输出镜像inspect；源码commit写入镜像label。
首次构建需要访问NGC、GitHub和Python依赖源，可能耗时较长；没有实现离线依赖镜像源或跨CPU仿真。
OpenPI与llama.cpp固定commit；模型完全固定SHA。Python依赖复用原deployment约束，
实际完整`pip freeze`保存在镜像`/opt/requirements-built.txt`。
尚未在Thor解析并验证完整transitive lock，不能声称不同日期重新build会逐字节相同。
需要长期复现时，应保存首次验收成功的镜像ID、基础digest、freeze和镜像tar。

镜像内隔离venv继承NGC Torch，将旧OpenPI依赖安装到venv；验证Torch版本未被替换。
安装官方固定OpenPI的`transformers_replace`，对AdaRMS做导入检查。
没有为Thor编译FlashRT，没有安装Orin的NVIDIA Torch2.3 wheel。

## 3. 逐步验收

```bash
# 无权重：必须识别SM110，并完成真实BF16 CUDA矩阵乘法。
bash containers/run.sh doctor

export ASSETS_DIR=/absolute/path/pi05-qwen-assets
export RESULTS_DIR=/absolute/path/thor-results

# 只验证挂载的五个模型文件。
bash containers/run.sh verify-assets

# 同时启动两个模型，检查HTTP/SSE/WS、图像改变和动态任务输入；完成后退出。
bash containers/run.sh smoke

# 原PyTorch baseline的完整策略、纯模型、内部vision/prefix/10轮去噪拆分，默认30次+10预热。
bash containers/run.sh baseline

# 双常驻空闲、仅VLA、仅VLM、持续并发、目标2Hz/0.2Hz等场景。
bash containers/run.sh benchmark
```

每次默认创建带UTC时间的新结果目录；可附`--output /results/custom-new-run`。
benchmark默认60s单模型场景、180s持续并发、30s空闲、60s指定频率，错过触发跳过。
VLA相机224×224、两路有效+一路masked、文本容量200、10×7输出；固定noise仅用于测量。
VLM两张512×384图片、context8192、单slot、最大96输出tokens、thinking/cache关闭，沿用原工作负载。
单独运行baseline和组合benchmark时，无需修改部署脚本或宿主机systemd。
请确保没有其他GPU推理负载；测试不会自动停止宿主机或其他容器的服务。

报告只在整个benchmark成功后生成；失败保留`completed:false`、原因、已测数据和日志。
VLA在同一实例内要求固定输出逐元素一致，跨GPU/后端不强制bitwise一致。
`smoke`是接口/输入响应检查，不是机器人任务成功率或精度验收。

## 4. 常驻服务

```bash
ASSETS_DIR=/absolute/path/pi05-qwen-assets RESULTS_DIR=/absolute/path/thor-results \
  bash containers/run.sh serve
```

只将宿主机127.0.0.1的8081/8082/8000映射到容器；运行时按请求推理，没有自动定时循环。
8081为VLA HTTP，8082为Qwen兼容API，8000为OpenPI msgpack WebSocket。
Ctrl+C/SIGTERM结束自己创建的模型子进程，保留外部结果文件。
没有使用`--privileged`、宿主机PID命名空间或systemd socket；不调用宿主机服务/功耗设置。
不要同时运行serve与benchmark占用同一GPU测量性能。

## 5. 内存、温度和功耗

`telemetry_container.py`每秒记录自己模型子进程的RSS/PSS、主机MemAvailable、
可读取hwmon原始值，以及`nvidia-smi`提供的设备内存、温度、功耗、活跃率和频率。
未知/不支持指标保留N/A和查询错误，不能用零冒充。
Thor设备是否实际暴露这些NVML/sensor指标需要验收；不预设与Orin的NvMap同义。
NVML设备功耗未被认定为Orin三轨合计或整机AC功耗，raw hwmon不盲目求和。
设备内存、RSS、系统内存有不同且可能重叠的范围，不相加。
1Hz采样可能漏掉瞬态峰值，固定测试顺序/动态时钟不构成长期热稳态认证。

跨平台比较应首先采用**Orin PyTorch baseline对Thor PyTorch baseline**。
Thor PyTorch与Orin FlashRT的比值同时混入后端差异，不能直接称作Thor硬件加速比。
FlashRT Thor移植应在此基线之后单独建分支、记录数值偏差和加速结果。

## 6. 打包和转移

本地从已提交的Git blob产生源码包，内容有逐文件SHA，不带模型或凭据：

```bash
python containers/package_buildkit.py --output /path/pi05-qwen-thor-buildkit.tar.gz
tar -xzf /path/pi05-qwen-thor-buildkit.tar.gz
cd pi05-qwen-thor-buildkit
```

Thor镜像验收成功后，才导出可搬运的运行镜像：

```bash
docker save pi05-qwen:thor-candidate -o pi05-qwen-thor-image.tar
# 另一台兼容Thor上：
docker load -i pi05-qwen-thor-image.tar
# 同时传输独立的pi05-qwen-assets目录；执行doctor、verify-assets、smoke后再测量。
```

CPU检查：`python -m unittest discover -s containers/tests -v`。
这些只验证校验/导出、打包范围、缺失值和报告统计，不代表Docker构建或GPU模型运行通过。
拿到Thor后的待办：匹配系统版本、真实build、GPU doctor、模型数值和接口检查、
两模型及监控完整测试、保存镜像/freeze/digest、另验证FlashRT支持范围。
