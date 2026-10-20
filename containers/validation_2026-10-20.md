# 构建包准备验证记录

日期：2026-10-20；目标：将来在Jetson Thor上构建和验收。

已通过：

- 5项CPU单元检查：权重导出/哈希/拒绝覆盖、包范围排除、原实测pin一致、缺失指标处理、预热排除和统计。
- 所有新增Python文件编译检查，runtime/assets CLI帮助可用。
- Git Bash对build.sh、run.sh、entrypoint.sh的`bash -n`语法检查。

尚未进行：

- Dockerfile真实构建、ARM64依赖解析、SM110编译与容器GPU运行。
- Thor双模型的数值一致性、HTTP/SSE/WS、延时、内存、温度与功耗测试。
- Thor传感器/NVML读数语义确认、完整依赖lock、FlashRT移植。

原因：目前没有可连接的Thor，本地没有Docker；当前Orin SSH也超时。
CPU检查不代表GPU通过，源码构建包不等于已构建/已验证的运行镜像。
