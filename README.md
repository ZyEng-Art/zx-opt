# zx-opt

ZX 图改写与单电路学习搜索研究。当前唯一搜索目标是 `barenco_tof_3`，优化提取和后处理后的 H/RZ/X/CX 总门数，CX 数作为同门数时的比较项。

本轮从已有 RL-ZX 单电路实验继续，加入受控图展开、相位拆分与重新分配、多个 gadget 的恒等改写，以及保留输入输出线序的结构去重。新增功能先验证等价性，再做同电路、同时间预算的消融实验。

已接入六类新动作：恒等展开、相位拆分、蜘蛛拆分、gadget 拆分、同目标集相位转移、spider-nest 补集替换。结构去重使用 WL 分桶和精确图同构，并记录每个结构状态已尝试的动作；代价缓存保留精确序列化键。

提取器对同构图的编号和布局敏感，因此另提供 `--dedup --dedup-scope serialized`：按精确表示保留动作记录和重启状态，避免跨表示误屏蔽。`--dedup --replace-better` 单独测试同构类内更优代表替换。`--frontier-reserve 512` 额外保存仍有未尝试动作的父图，在主状态池耗尽时恢复分支；主池保持 64，储备容量单独报告。

默认主池和成本缓存一直使用精确序列化键；`--dedup` 另外启用全局已尝试动作屏蔽，并选择同构或精确表示粒度。同构关系始终单独统计。

从仓库目录运行：

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m zxopt.search runs/example --expanded --dedup --policy online --seconds 180
```

复现有限分支储备的在线组（旧规则，600 秒）：

```sh
.venv/bin/python -m zxopt.search runs/frontier --dedup --dedup-scope serialized --frontier-reserve 512 --policy online --seconds 600
```

对应的随机、冻结和在线完整对照用 `python -m zxopt.campaign runs/frontier-study --suite frontier --seconds 600 --workers 3`。冻结组与在线组从相同的随机初始化开始，不使用预训练检查点。

完整命令与输出说明见 [docs/runbook.txt](docs/runbook.txt)，已实现的规则和剩余缺口见 [docs/rule-coverage.txt](docs/rule-coverage.txt)。

当前实现不声明 ZX 规则完备，也不把损失下降视为门数优化有效。完整规则系的公理覆盖、相位范围和 flow 条件将单独记录。

首轮已完成 6 组 × 3 种子 × 180 秒的同电路比较：旧动作随机组为 36/36/37 门，扩展动作加去重的随机组为 38/36/37 门，在线组为 38/37/37 门。目前没有稳定的在线学习收益。详见[完整解读](reports/pilot-20261017/interpretation.txt)、[搜索曲线](reports/pilot-20261017/anytime.png)与[学习诊断](reports/pilot-20261017/learning.png)。

保存的 24,669 份 QASM 已通过固定线序全矩阵复核。运行使用 node44 的已有环境，实际版本和证据摘要保存在 [reports/pilot-20261017](reports/pilot-20261017)；本地开发依赖列在 `requirements.txt`。

后续完成了 18 次表示粒度对照、2 次本地储备预检和 12 次 node44 储备对照。表示对照发现去重组会提前耗尽保留池；新增储备使完整对照中的 9 次启用运行均用满 600 秒。精确去重随机组为 37/37/37 门，增加储备后为 37/37/36 门；相同储备下的冻结和在线组均为 36/36/36 门，但在线组平均搜索曲线较差。长训确实形成动作偏好，尚未显示稳定的学习收益，整体最好仍为 36 门。详见[表示对照](reports/representation-20261017/interpretation.txt)、[配对预检](reports/frontier-local-20261017/interpretation.txt)和[完整储备对照](reports/frontier-20261017/interpretation.txt)。

开发计划和边界见 [docs/plan.txt](docs/plan.txt)，逐步记录见 [docs/work-log.jsonl](docs/work-log.jsonl)。原始运行数据与模型保存在 `runs/`，Git 记录配置、源码摘要、结果摘要和可复现命令。

参考：

- [Circopt-RL-ZXCalc](https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc)
- [Quarl](https://github.com/quantum-compiler/Quarl)
- [Clifford+T 与一般 ZX 的完备性](https://arxiv.org/abs/1903.06035)
- [Flow-preserving 完备规则](https://arxiv.org/abs/2608.13035)
