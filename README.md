# zx-opt

ZX 图改写与单电路学习搜索研究。当前唯一搜索目标是 `barenco_tof_3`，优化提取和后处理后的 H/RZ/X/CX 总门数，CX 数作为同门数时的比较项。

本轮从已有 RL-ZX 单电路实验继续，加入受控图展开、相位拆分与重新分配、多个 gadget 的恒等改写，以及保留输入输出线序的结构去重。新增功能先验证等价性，再做同电路、同时间预算的消融实验。

已接入六类新动作：恒等展开、相位拆分、蜘蛛拆分、gadget 拆分、同目标集相位转移、spider-nest 补集替换。结构去重使用 WL 分桶和精确图同构，并记录每个结构状态已尝试的动作；代价缓存保留精确序列化键。

提取器对同构图的编号和布局敏感，因此另提供 `--dedup --dedup-scope serialized`：按精确表示保留动作记录和重启状态，避免跨表示误屏蔽。`--dedup --replace-better` 单独测试同构类内更优代表替换。`--suite representation` 对两者及冻结/在线策略做旧规则对照。

从仓库目录运行：

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m zxopt.search runs/example --expanded --dedup --policy online --seconds 180
```

完整命令与输出说明见 [docs/runbook.txt](docs/runbook.txt)，已实现的规则和剩余缺口见 [docs/rule-coverage.txt](docs/rule-coverage.txt)。

当前实现不声明 ZX 规则完备，也不把损失下降视为门数优化有效。完整规则系的公理覆盖、相位范围和 flow 条件将单独记录。

首轮已完成 6 组 × 3 种子 × 180 秒的同电路比较：旧动作随机组为 36/36/37 门，扩展动作加去重的随机组为 38/36/37 门，在线组为 38/37/37 门。目前没有稳定的在线学习收益。详见[完整解读](reports/pilot-20261017/interpretation.txt)、[搜索曲线](reports/pilot-20261017/anytime.png)与[学习诊断](reports/pilot-20261017/learning.png)。

保存的 24,669 份 QASM 已通过固定线序全矩阵复核。运行使用 node44 的已有环境，实际版本和证据摘要保存在 [reports/pilot-20261017](reports/pilot-20261017)；本地开发依赖列在 `requirements.txt`。

开发计划和边界见 [docs/plan.txt](docs/plan.txt)，逐步记录见 [docs/work-log.jsonl](docs/work-log.jsonl)。原始运行数据与模型保存在 `runs/`，Git 记录配置、源码摘要、结果摘要和可复现命令。

参考：

- [Circopt-RL-ZXCalc](https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc)
- [Quarl](https://github.com/quantum-compiler/Quarl)
- [Clifford+T 与一般 ZX 的完备性](https://arxiv.org/abs/1903.06035)
- [Flow-preserving 完备规则](https://arxiv.org/abs/2608.13035)
