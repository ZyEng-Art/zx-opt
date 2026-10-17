# zx-opt

ZX 图改写与逐电路学习搜索研究。每条电路独立从头训练和搜索，优化提取和后处理后的 H/RZ/X/CX 总门数，CX 数作为同门数时的比较项。已有 `barenco_tof_3` 的多轮对照，并已支持 `tof_3`、`mod5_4` 和 `tof_4` 的独立实验。

当前入口 `zxopt.general_search` 已将 JPV 的 11 类 Clifford+T 公理接入一般 ZX 图搜索和 PPO。状态保留红绿蜘蛛、H、平行线与标量；评分在独立副本进行。合法但不可提取的中间图可以继续改写，进入单独的潜在图池，真实成本保留为未知。策略观察一般图、评分图与动作，目标仍是提取后总 NAM 门数。

数学内核支持完整公理模板的双向上下文改写；训练采用有限候选、匹配工作量、图大小和轨迹上限，因此不声称完备搜索或门数全局最优。范围是 Clifford+T，不包括任意角度的 ZX 完备性。规则覆盖和精确验证见 [完备性说明](docs/clifford-t-completeness.txt)与[内核报告](reports/clifford-t-20261017/interpretation.txt)。

一般图搜索可用 `--allow-arbitrary-angles` 接受混合角度输入：保留非 π/4 相位，仅跳过不满足当前 JPV 模板角度条件的匹配。全算子与输出电路核验保持启用；这不代表任意角度规则完备。用法与数值表示边界见[任意角度输入说明](docs/arbitrary-angles.txt)。

一般图训练先过滤实际子图超出资源预算的动作，过滤后的候选集同时用于采样和 PPO 概率重算；过滤不做提取或查看成本，耗时全部计入预算。错误提取电路被隔离，不参与奖励或最优结果。具体流程、池预算及对照口径见 [一般图搜索说明](docs/general-search.txt)。

原 `zxopt.search` 保留已有 RL-ZX 单电路实验、受控展开和结构去重，用于复现历史对照：

已接入六类新动作：恒等展开、相位拆分、蜘蛛拆分、gadget 拆分、同目标集相位转移、spider-nest 补集替换。结构去重使用 WL 分桶和精确图同构，并记录每个结构状态已尝试的动作；代价缓存保留精确序列化键。

提取器对同构图的编号和布局敏感，因此另提供 `--dedup --dedup-scope serialized`：按精确表示保留动作记录和重启状态，避免跨表示误屏蔽。`--dedup --replace-better` 单独测试同构类内更优代表替换。`--frontier-reserve 512` 额外保存仍有未尝试动作的父图，在主状态池耗尽时恢复分支；主池保持 64，储备容量单独报告。

默认主池和成本缓存一直使用精确序列化键；`--dedup` 另外启用全局已尝试动作屏蔽，并选择同构或精确表示粒度。同构关系始终单独统计。

从仓库目录运行：

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m zxopt.general_search runs/example --jpv --policy online --seconds 1200
```

六组随机/冻结/在线对照分别启用和关闭 JPV，其余网络、评分和池相同：

```sh
.venv/bin/python -m zxopt.general_campaign runs/jpv-study --seconds 1200 --workers 3
.venv/bin/python tools/audit_general.py runs/jpv-study runs/jpv-audit --workers 3
```

复现有限分支储备的在线组（旧规则，600 秒）：

```sh
.venv/bin/python -m zxopt.search runs/frontier --dedup --dedup-scope serialized --frontier-reserve 512 --policy online --seconds 600
```

对应的随机、冻结和在线完整对照用 `python -m zxopt.campaign runs/frontier-study --suite frontier --seconds 600 --workers 3`。冻结组与在线组从相同的随机初始化开始，不使用预训练检查点。

完整命令与输出说明见 [docs/runbook.txt](docs/runbook.txt)，已实现的规则和剩余缺口见 [docs/rule-coverage.txt](docs/rule-coverage.txt)。

损失下降和策略分布变化用于诊断训练；是否改善搜索要看相同预算下随机、冻结和在线的全部种子结果。

其他电路对照已完成：`tof_3`、`mod5_4`、`tof_4` 各四方法×三种子×1200秒，共36次，全部独立从头训练。共同预处理后的35/41/55门分别降到33/22/52门；这些最好成绩宏动作随机组也能达到。JPV在线相对冻结的最终成本为3胜5平1负，三条平均搜索曲线略好，但策略变化小，尚未证明稳定学习收益。98,954次接受改写和39,190份电路核验通过。详见[完整报告](reports/other-circuits-20261017/interpretation.txt)、[mod5_4曲线](reports/other-circuits-20261017/mod5_4/anytime.png)、[tof_4曲线](reports/other-circuits-20261017/tof_4/anytime.png)和[复现配置](docs/other-circuits.txt)。

一般图 JPV 正式对照已完成：六组 × 三种子 × 1200 秒，最好仍为 36 门/14 CX。含 JPV 的在线组为 37/37/37 门，冻结组为 36/38/37 门，随机组为 38/36/37 门；value loss 下降和策略偏好变化尚未带来稳定搜索收益。159,998 次接受改写及 47,814 份电路全部通过重放和固定线序核验。原控制器在后续源码审计遇到 Git 所有权检查而退出，失败回执保留，源码审计和策略探针已在本地补完。详见[完整报告](reports/jpv-20261017/interpretation.txt)、[搜索曲线](reports/jpv-20261017/anytime.png)、[学习曲线](reports/jpv-20261017/learning.png)及[收尾回执](reports/jpv-20261017/postprocessing.json)。

一般图接入后的资源过滤诊断已完成：种子 29，冻结/在线分别启用和关闭 JPV，四组各 600 秒。全部 4,644 次接受改写和 1,885 份电路通过核验，资源超限拒绝为零；过滤仍占约 69%～71% 的归类耗时。JPV 两组均为 37 门，宏动作两组均为 38 门，单种子诊断未显示在线优势。详见[修复复测](reports/jpv-mask-20261017/interpretation.txt)。

首轮已完成 6 组 × 3 种子 × 180 秒的同电路比较：旧动作随机组为 36/36/37 门，扩展动作加去重的随机组为 38/36/37 门，在线组为 38/37/37 门。目前没有稳定的在线学习收益。详见[完整解读](reports/pilot-20261017/interpretation.txt)、[搜索曲线](reports/pilot-20261017/anytime.png)与[学习诊断](reports/pilot-20261017/learning.png)。

保存的 24,669 份 QASM 已通过固定线序全矩阵复核。运行使用 node44 的已有环境，实际版本和证据摘要保存在 [reports/pilot-20261017](reports/pilot-20261017)；本地开发依赖列在 `requirements.txt`。

后续完成了 18 次表示粒度对照、2 次本地储备预检和 12 次 node44 储备对照。表示对照发现去重组会提前耗尽保留池；新增储备使完整对照中的 9 次启用运行均用满 600 秒。精确去重随机组为 37/37/37 门，增加储备后为 37/37/36 门；相同储备下的冻结和在线组均为 36/36/36 门，但在线组平均搜索曲线较差。长训确实形成动作偏好，尚未显示稳定的学习收益，整体最好仍为 36 门。详见[表示对照](reports/representation-20261017/interpretation.txt)、[配对预检](reports/frontier-local-20261017/interpretation.txt)和[完整储备对照](reports/frontier-20261017/interpretation.txt)。

开发计划和边界见 [docs/plan.txt](docs/plan.txt)，逐步记录见 [docs/work-log.jsonl](docs/work-log.jsonl)。原始运行数据与模型保存在 `runs/`，Git 记录配置、源码摘要、结果摘要和可复现命令。

参考：

- [Circopt-RL-ZXCalc](https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc)
- [Quarl](https://github.com/quantum-compiler/Quarl)
- [Clifford+T 与一般 ZX 的完备性](https://arxiv.org/abs/1903.06035)
- [Flow-preserving 完备规则](https://arxiv.org/abs/2608.13035)
