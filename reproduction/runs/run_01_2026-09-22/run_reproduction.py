#!/usr/bin/env python3
"""One-command computational reproduction. Expected author failures are retained."""
import argparse,sys,subprocess,datetime,json
from pathlib import Path
REP=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-esm',action='store_true',help='reuse verified recomputed embeddings')
    p.add_argument('--skip-notebooks',action='store_true',help='reuse notebook execution evidence')
    a=p.parse_args();sys.path.insert(0,str(REP/'src'))
    from common import ROOT
    steps=[('data_audit',['src/audit.py'])]
    if not a.skip_notebooks:steps += [('original_notebooks',['src/original_notebooks.py']),('path_repaired_notebooks',['src/repaired_notebooks.py'])]
    steps += [('embeddings',['src/embeddings.py']+(['--skip-esm'] if a.skip_esm else [])),
        ('cleanlab',['src/cleaning.py']),('cleanlab_sensitivity',['src/cleanlab_sensitivity.py']),
        ('models',['src/models.py','--models','SVM','LR','KNN','RF','LightGBM']),
        ('svm_experiments',['src/models.py','--svm-experiments']),('paper_parameters',['src/paper_parameters.py']),('lstm',['src/lstm.py']),
        ('leakage',['src/leakage.py']),('report',['src/report.py']),('verification',['src/verify.py'])]
    records=[]
    for name,args in steps:
        print(f'START {name}',flush=True)
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (REP/'logs'/f'pipeline_{name}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(REP/args[0]),*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(stage=name,exit_code=r.returncode,start_utc=start))
        (REP/'results/pipeline_status.json').write_text(json.dumps(records,indent=2))
        print(f'{"DONE" if r.returncode==0 else "FAILED"}: {name}; see logs/pipeline_{name}.log',flush=True)
    # Expected original notebook failures are described in their status CSV, not hidden.
    if any(x['exit_code'] for x in records):raise SystemExit(1)
if __name__=='__main__':main()
