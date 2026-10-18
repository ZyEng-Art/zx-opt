"""Join Quarl Table 2 with one consistent, source-bound RL-ZX NAM cohort."""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["Circuit", "Orig.", "Nam", "VOQC", "Qiskit", "Tket", "Quartz w/ RM",
           "Quartz w/o RM", "Quarl w/ RM", "Quarl w/o RM", "RL-ZX (original cohort)"]
COMPACT = [0, 1, 2, 3, 4, 5, 6, 8, 10]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TableParser(HTMLParser):
    """Ignore nested header tables while retaining the outer table's cells."""
    def __init__(self):
        super().__init__(); self.rows = []; self.row = self.cell = None; self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table": self.depth += 1
        if tag == "tr" and self.depth == 1: self.row = []
        if tag in ("td", "th") and self.depth == 1: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.depth == 1 and self.cell is not None:
            self.row.append(re.sub(r"\s+", " ", " ".join(self.cell)).strip()); self.cell = None
        if tag == "tr" and self.depth == 1 and self.row:
            self.rows.append(self.row); self.row = None
        if tag == "table": self.depth -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    section = re.search(r'<figure id="S6.T2".*?</figure>', args.html.read_text(), re.S)
    if section is None: raise ValueError("Missing exact Table 2")
    parsed = TableParser(); parsed.feed(section.group())
    assert len(parsed.rows) == 29 and len(parsed.rows[1]) == 10
    manifest_path = ROOT/"docs/quarl-nam-inputs.json"
    manifest = json.loads(manifest_path.read_text()); cases = {r["name"]: r for r in manifest["cases"]}
    originals = ROOT/"reports/anchor-first-20261018/original-nam26"
    records = {}; source_files = []
    for node in (22, 23):
        path = originals/("node"+str(node)+".json"); source_files.append(path)
        for record in json.loads(path.read_text())["records"]:
            name = record["circuit"]; assert name not in records
            p, r = record["protocol"], record["report"]
            assert p["git_commit"] == "f5874e497ce17b6bacddadb8e907763a9eb299f6" and not p["tracked_dirty"]
            assert p["config"]["seed"] == 11 and p["config"]["seconds"] == 0
            assert p["config"]["no_improvement_seconds"] == 10800
            assert r["status"] == "no_improvement_timeout" and r["verification"] == "passed"
            assert r["finished_unix"]-r["patience"]["last_improvement_unix"] >= 10800
            assert p["input_sha256"] == cases[name]["raw"]["sha256"]
            records[name] = record
    assert len(records) == len(cases) == 26
    joined = []; bindings = []
    for raw in parsed.rows[2:-1]:
        assert len(raw) == 10
        name = raw[0].replace(" ", "").replace("ˆ", "^")
        values = [None if value == "-" else int(value) for value in raw[1:]]
        case = cases[name]; own = records[name]; report = own["report"]
        assert values[0] == case["paper_original_gates"] == own["protocol"]["input_gates"]
        assert values[7] == case["paper_with_rm"]["gates"]
        assert values[8] == case["paper_without_rm"]["gates"]
        joined.append([name]+values+[report["best"]["cost"]["gates"]])
        bindings.append(dict(circuit=name, input_sha256=own["protocol"]["input_sha256"],
                             report_sha256=own["report_sha256"], protocol_sha256=own["protocol_sha256"],
                             best_qasm_sha256=own["best_qasm_sha256"],
                             root_gates=report["root_cost"]["gates"], final=report["best"]["cost"],
                             seconds_after_last_improvement=report["finished_unix"]-report["patience"]["last_improvement_unix"]))
    assert {r[0] for r in joined} == set(cases)
    published = parsed.rows[-1]
    for index in range(2, 10):
        ratios = [math.log(r[index]/r[1]) for r in joined if r[index] is not None]
        recomputed = 100*(1-math.exp(sum(ratios)/len(ratios)))
        assert round(recomputed, 1) == float(published[index].rstrip("%"))
    own_reduction = 100*(1-math.exp(sum(math.log(r[-1]/r[1]) for r in joined)/26))
    geo = ["Geo. mean reduction", None]+published[2:]+[f"{own_reduction:.1f}%"]
    for filename, indices in (("table-full.csv", list(range(11))), ("table.csv", COMPACT)):
        with (out/filename).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([COLUMNS[i] for i in indices])
            writer.writerows([["-" if row[i] is None else row[i] for i in indices] for row in joined+[geo]])
    lines = [" | ".join(COLUMNS[i] for i in COMPACT), " | ".join(["---"]+["---:"]*8)]
    for row in joined+[geo]:
        best = min(row[i] for i in COMPACT[2:] if isinstance(row[i], int)) if row is not geo else None
        text = []
        for i in COMPACT:
            value = "-" if row[i] is None else str(row[i])
            if i >= 2 and row[i] == best: value = "**"+value+"**"
            text.append(value)
        lines.append(" | ".join(text))
    (out/"table.txt").write_text("\n".join(lines)+"\n")
    outcomes = {column: dict(Counter("win" if r[-1] < r[column] else "loss" if r[-1] > r[column] else "tie"
                                     for r in joined)) for column in (8, 9)}
    result = dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        paper="https://arxiv.org/html/2307.10120v1#S6.T2", table=2, columns=COLUMNS, rows=joined,
        geo_mean_reduction=geo, own_geo_reduction_percent=own_reduction, outcomes_vs_quarl=outcomes,
        own_cohort=dict(source_commit="f5874e497ce17b6bacddadb8e907763a9eb299f6", seed=11,
                       stop="Individual3hours without strict total-gate improvement; no overall cap",
                       latest_anchor_first=False, multi_seed_best_mixed=False),
        bindings=bindings, sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files+[manifest_path]},
        fetched_html_sha256=sha(args.html), generator_sha256=sha(Path(__file__)),
        checks="All26inputSHA/originalcost/QuarlRM/noRM counts and published geometric means crosschecked",
        caveats=["Paper baselines are published values, not locally rerun methods",
                 "Runtime/hardware/pretraining/extraction/postprocessing differ; this is a final gate-count comparison",
                 "Own column uses one consistent original26cohort; barenco_tof_3 remains37, later36 is excluded",
                 "Own counts include deterministic preprocessing, so reductions are not isolated RL benefits",
                 "Runtime equivalence checks passed; no new full-trajectory independent audit claimed for original26",
                 "Nam geometric mean uses24reported circuits; other columns use26"])
    (out/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42})
    labels = ["Circuit", "Orig.", "Nam", "VOQC", "Qiskit", "Tket", "Quartz\n+RM", "Quartz\nno RM", "Quarl\n+RM", "Quarl\nno RM", "Our RL-ZX\n(old)"]
    celltext = [["-" if v is None else str(v) for v in row] for row in joined+[geo]]
    fig, ax = plt.subplots(figsize=(16.5, 11.7)); ax.axis("off")
    table = ax.table(cellText=celltext, colLabels=labels, colWidths=[.19]+[.081]*10,
                     cellLoc="right", loc="center", bbox=[0, .14, 1, .79])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db"); cell.set_linewidth(.4)
        if ri == 0: cell.set_facecolor("#e5e7eb"); cell.get_text().set_weight("bold")
        elif ci == 10: cell.set_facecolor("#e0f2fe")
        if ci == 0: cell.get_text().set_ha("left")
    for ri, row in enumerate(joined, 1):
        best = min(v for v in row[2:] if v is not None)
        for ci in range(2, 11):
            if row[ci] == best: table[ri, ci].get_text().set_weight("bold")
    fig.suptitle("NAM total gate count: Quarl Table 2 with our RL-ZX results appended", fontsize=16, y=.97)
    ax.text(0, .96, "Lower is better. Bold = row minimum across reported optimizers. Blue = our consistent26circuit cohort.", transform=ax.transAxes, fontsize=10)
    notes = ("Paper: https://arxiv.org/html/2307.10120v1#S6.T2  |  RM = rotation merging; '-' = unreported.\n"
             "Our column: source f5874e4, seed11, each circuit stops after3hours without a new total-gate record.\n"
             "Includes our extraction/postprocessing; not the latest anchor-first scheme. Later barenco36gate results are excluded.\n"
             "Published baseline runs and our runs differ in compute/training budget. Gate-count reductions do not isolate RL utility.\n"
             f"Our geometric mean reduction: {own_reduction:.2f}% over26circuits. Nam's published geometric mean excludes2missing rows.")
    ax.text(0, .025, notes, transform=ax.transAxes, fontsize=9, linespacing=1.6)
    fig.savefig(out/"table.pdf", bbox_inches="tight"); fig.savefig(out/"table.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(dict(circuits=26, own_geo_reduction=own_reduction, outcomes=outcomes)))


if __name__ == "__main__":
    main()
