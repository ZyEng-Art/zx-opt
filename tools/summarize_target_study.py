"""Per-target reports; never pool absolute gate counts across different circuits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path); p.add_argument("audit", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.study/"campaign.json").read_text())
    verification = json.loads((args.audit/"verification.json").read_text())
    assert verification["status"] == "passed"
    targets = []; pairs = []; lines = ["其他电路独立训练搜索，所有目标/方法/种子均保留。", "",
        "每条电路从原始输入、空状态池和随机网络开始；冻结/在线同初始化。",
        "没有迁移 barenco 的模型，不用本轮结果声称跨电路泛化。", "",
        "目标 | 输入门数 | 预处理基线 | 方法 | 最终门数（各个种子） | 时间平均最低门数"]
    for target in manifest["targets"]:
        name = target["circuit"]; folder = args.study/name; out = args.output/name
        subprocess.run([sys.executable, str(ROOT/"tools/summarize_general.py"), str(folder), str(out)], check=True)
        for filename in ["verification.json", "source-audit.json", "policy-probe.json"]:
            source = args.audit/name/filename
            assert json.loads(source.read_text())["status"] == "passed"
            shutil.copyfile(source, out/filename)
        shutil.copyfile(folder/"campaign.json", out/"campaign.json")
        a = json.loads((out/"analysis.json").read_text())
        first = a["trials"][0]
        protocol = json.loads((folder/first["folder"]/"protocol.json").read_text())
        baseline = protocol["root_cost"]["gates"]
        records = []
        for row in a["aggregate"]:
            lines.append(f"{name} | {protocol['input_gates']} | {baseline} | {row['arm']} | {row['final_gates']} | {row['mean_anytime_gates']:.6f}")
            records.append(dict(row, mean_final_fraction_of_baseline=sum(row["final_gates"])/(len(row["final_gates"])*baseline)))
        for row in a["paired"]:
            if row["base"] != "jpv":
                continue
            def cost(key):
                v = row[key]; return (v["gates"], v["cx"])
            online, frozen = cost("online_final"), cost("frozen_final")
            pairs.append(dict(target=name, **row, result="win" if online < frozen else "loss" if online > frozen else "tie"))
        targets.append(dict(circuit=name, num_qubits=protocol["num_qubits"], input_gates=protocol["input_gates"],
                            input_sha256=protocol["input_sha256"], baseline=protocol["root_cost"], arms=records,
                            experiment_commit=protocol["git_commit"],
                            online_diagnostics=[dict(seed=r["seed"], updates=r["updates"],
                                learning=r.get("learning"), feasibility=r.get("feasibility"), timers=r["timers"])
                                for r in a["trials"] if r["arm"] == "jpv_online"]))
    counts = {key: sum(p["result"] == key for p in pairs) for key in ["win", "tie", "loss"]}
    result = dict(status="passed", targets=targets, paired_online_vs_frozen=pairs, paired_final_outcomes=counts,
                  budget_seconds=manifest["seconds"], seeds=manifest["seeds"],
                  scope="Independent on-target training; exploratory per-target comparisons, no cross-target model transfer")
    (args.output/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    shutil.copyfile(args.audit/"verification.json", args.output/"verification.json")
    shutil.copyfile(args.study/"campaign.json", args.output/"campaign.json")
    lines.extend(["", f"在线相对冻结最终成本（总门数优先、CX 次之）：{counts}。",
                  "跨电路的绝对门数不合并平均；每个电路的曲线、诊断、全部种子和审计在对应子目录。",
                  "共同预处理收益与后续搜索收益分开；value loss 下降不单独证明学习有效。",
                  f"共重放 {verification['accepted_rewrites']} 次接受改写，核验 {verification['saved_qasm']} 份保存电路。"])
    (args.output/"findings.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="passed", targets=len(targets), paired_final_outcomes=counts)))


if __name__ == "__main__":
    main()
