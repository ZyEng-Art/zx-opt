# zx-opt

ZX 图改写与单电路学习搜索研究。当前唯一搜索目标是 `barenco_tof_3`，优化提取和后处理后的 H/RZ/X/CX 总门数，CX 数作为同门数时的比较项。

本轮从已有 RL-ZX 单电路实验继续，加入受控图展开、相位拆分与重新分配、多个 gadget 的恒等改写，以及保留输入输出线序的结构去重。新增功能先验证等价性，再做同电路、同时间预算的消融实验。

当前实现不声明 ZX 规则完备，也不把损失下降视为门数优化有效。完整规则系的公理覆盖、相位范围和 flow 条件将单独记录。

开发计划和边界见 [docs/plan.txt](docs/plan.txt)，逐步记录见 [docs/work-log.jsonl](docs/work-log.jsonl)。原始运行数据与模型保存在 `runs/`，Git 记录配置、源码摘要、结果摘要和可复现命令。

参考：

- [Circopt-RL-ZXCalc](https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc)
- [Quarl](https://github.com/quantum-compiler/Quarl)
- [Clifford+T 与一般 ZX 的完备性](https://arxiv.org/abs/1903.06035)
- [Flow-preserving 完备规则](https://arxiv.org/abs/2608.13035)
