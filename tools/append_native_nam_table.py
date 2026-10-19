"""Append the source-bound native NAM cohort to the existing Quarl comparison."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT/'reports/nam-results-table-20261018/table-full.csv'
    cohort = ROOT/'reports/native-nam-results-20261019'
    out = ROOT/'reports/nam-results-table-20261019'
    out.mkdir(exist_ok=True)
    with old.open() as f:
        table = list(csv.reader(f))
    with (cohort/'results.csv').open() as f:
        results = {r['circuit']: r for r in csv.DictReader(f)}
    assert len(results) == 26
    columns = table[0]+['RL-ZX native (2026-10-19, audit pending)']
    rows = []; bindings = []
    for row in table[1:-1]:
        name = row[0]; own = results[name]
        d = cohort/('node'+own['node'])/'comparison'/name/'rules_native_online_s11'
        p = json.loads((d/'protocol.json').read_text())
        r = json.loads((d/'report.json').read_text())
        assert int(row[1]) == int(own['input_gates']) == p['input_gates']
        assert p['git_commit'] == '26b4e8eff8625219c253820a7ad0a6903c18bdca' and not p['tracked_dirty']
        assert p['active_rules']['profile']=='native' and not p['active_rules']['macros'] and not p['active_rules']['jpv']
        assert p['config']['seed']==11 and p['config']['policy']=='online'
        assert p['stop_rule']['overall_cap_seconds'] is None
        assert r['status']=='no_improvement_timeout' and r['finished_unix']-r['patience']['last_improvement_unix']>=10800
        assert int(own['best_gates'])==r['best']['cost']['gates']
        assert hashlib.sha256((d/'best.qasm').read_bytes()).hexdigest()==r['best']['sha256']
        rows.append(row+[own['best_gates']])
        bindings.append(dict(circuit=name,report_sha256=hashlib.sha256((d/'report.json').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((d/'protocol.json').read_bytes()).hexdigest(),best_qasm_sha256=r['best']['sha256']))
    assert {r[0] for r in rows} == set(results)
    geo = 100*(1-math.exp(sum(math.log(int(r[-1])/int(r[1])) for r in rows)/26))
    footer = table[-1]+[f'{geo:.1f}%']
    for name, indices in [('table-full.csv',list(range(len(columns)))),('table.csv',[0,1,8,9,10,11])]:
        with (out/name).open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow([columns[i] for i in indices]);w.writerows([[r[i] for i in indices] for r in rows+[footer]])
    text=[' | '.join(columns),' | '.join(['---']+['---:']*(len(columns)-1))]
    text += [' | '.join(row) for row in rows+[footer]]
    (out/'table.txt').write_text('\n'.join(text)+'\n')
    outcomes={columns[i]:dict(Counter('win' if int(r[-1])<int(r[i]) else 'loss' if int(r[-1])>int(r[i]) else 'tie' for r in rows)) for i in [8,9,10]}
    summary=dict(source_table=str(old.relative_to(ROOT)),source_table_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),columns=columns,rows=rows,geometric_mean_reduction_percent=geo,outcomes=outcomes,bindings=bindings,independent_audit_status='pending cohort completion; snapshot status retained',source_commit='26b4e8eff8625219c253820a7ad0a6903c18bdca',caveats=['Published paper columns are unchanged; not rerun on current hardware.','Old RL-ZX cohort remains unchanged; new native cohort is the last column.','One online seed11 per circuit; strict 3h gate plateau, no overall cap.','Includes deterministic preprocessing/extraction/postprocessing; not isolated learning benefit.','Total-sum reduction and geometric mean reduction are different metrics.'])
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42})
    labels=['Circuit','Orig.','Nam','VOQC','Qiskit','Tket','Quartz\n+RM','Quartz\nno RM','Quarl\n+RM','Quarl\nno RM','RL-ZX\nprevious','RL-ZX native\nnew / pending']
    fig,ax=plt.subplots(figsize=(19,12));ax.axis('off')
    t=ax.table(cellText=rows+[footer],colLabels=labels,colWidths=[.175]+[.075]*11,cellLoc='right',bbox=[0,.17,1,.78])
    t.auto_set_font_size(False);t.set_fontsize(9.5)
    for (ri,ci),cell in t.get_celld().items():
        cell.set_edgecolor('#d1d5db');cell.set_linewidth(.4)
        if ri==0:cell.set_facecolor('#e5e7eb');cell.get_text().set_weight('bold')
        elif ci==11:cell.set_facecolor('#dbeafe')
        if ci==0:cell.get_text().set_ha('left')
    for ri,row in enumerate(rows,1):
        best=min(int(v) for v in row[2:] if v!='-')
        for ci in range(2,12):
            if row[ci]!='-' and int(row[ci])==best:t[ri,ci].get_text().set_weight('bold')
    fig.suptitle('Quarl Table 2: NAM total gates with native RL-ZX results in the last column',fontsize=16,y=.97)
    ax.text(0,.08,'Lower is better. Blue = new native-only cohort; independent replay audit still in progress.\nPrevious paper / RL-ZX columns preserved. Seed 11; each target stops after 3 h without a strict gate improvement.\nPreprocessing, extraction, hardware and training budgets differ; counts do not isolate learning benefit.\nSource: arxiv.org/html/2307.10120v1#S6.T2. RM = rotation merging. Bold = row minimum across optimizers.',transform=ax.transAxes,fontsize=10,linespacing=1.5)
    fig.savefig(out/'table.pdf',bbox_inches='tight');fig.savefig(out/'table.png',dpi=140,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(geo=geo,outcomes=outcomes)))


if __name__=='__main__':
    main()
