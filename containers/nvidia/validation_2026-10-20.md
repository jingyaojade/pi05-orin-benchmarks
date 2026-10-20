# Linux RTX构建包准备检查

日期：2026-10-20；源码分支：`docker-nvidia`；目标：Linux x86_64 RTX4090/5090。
原Thor分支和已经导出的Thor归档保持不变。

已执行并通过：

- `python -m unittest discover -s containers/tests -v`：9项CPU检查。
  检查权重SHA/导出及拒绝覆盖、打包范围、原实测pin一致、Thor/RTX模型及上游pin一致、
  GPU可见序号/UUID集合重映射、无效设备拒绝、多卡指标分别统计、缺失值、预热排除和mean/p95。
- `containers`内8个Python文件AST解析，两份Dockerfile中的Python heredoc解析。
- Git Bash对所有容器shell脚本执行 `bash -n`，runtime和打包CLI帮助检查。
- `git diff --check`；代码范围凭据模式扫描未发现token、私钥或密码赋值。

CPU统计测试中的GPU读数是明确标注的单元测试fixture，仅在临时目录使用；没有作为实测报告交付。
Dockerfile语法/依赖/编译的完整正确性仍需真实Docker构建确认，AST与bash语法检查不能替代它。

尚未执行：

- 拉取NGC镜像、Docker build、x86依赖解析、SM89/SM120的llama.cpp编译。
- RTX CUDA doctor、两个模型运行、接口/数值检查、实际单卡或双卡基准。
- RTX延时、显存峰值、温度、设备功耗以及传感器/设备UUID关联的硬件验收。
- 完整transitive依赖锁、docker save镜像导出、RTX FlashRT移植。

原因：本地无Docker CLI/daemon，没有可连接的目标RTX Linux机器。
这是源码构建包，包含模型导出工具与固定SHA，**不包含权重或已经构建的运行镜像**。
后续必须按README的doctor→verify-assets→smoke→baseline→benchmark顺序在目标设备验收。
