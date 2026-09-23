"""Archive run 1 and execute the unchanged scientific pipeline in a fresh run directory."""
from pathlib import Path
import argparse,datetime,difflib,hashlib,json,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reproduction'
RUNS=REP/'runs'
BASE=RUNS/'run_01_2026-09-22'
SECOND=RUNS/'run_02_2026-09-23'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot():
    if BASE.exists():raise RuntimeError('Run 1 archive already exists; do not overwrite evidence')
    BASE.mkdir(parents=True)
    for name in ['results','models','data_audit','executed_notebooks','logs','sources','src']:
        shutil.copytree(REP/name,BASE/name,ignore=shutil.ignore_patterns('__pycache__','jupyter_runtime','git_final_state.txt','original_workdir'))
    for name in ['requirements.txt','run_reproduction.py','predict.py','REPRODUCTION_REPORT.md','DATA_AUDIT.md','ESM2_IMPLEMENTATION.md','repository_inventory.csv','repository_tree.txt']:
        shutil.copy2(REP/name,BASE/name)
    manifest=[]
    for p in sorted(BASE.rglob('*')):
        if p.is_file():manifest.append(dict(path=str(p.relative_to(BASE)),sha256=digest(p),size_bytes=p.stat().st_size))
    (BASE/'archive_manifest.json').write_text(json.dumps(dict(original_run_date='2026-09-22',archived_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest),indent=2)+'\n')
    return manifest

def prepare():
    if SECOND.exists():raise RuntimeError('Run 2 directory already exists; do not overwrite evidence')
    manifest=snapshot()
    SECOND.mkdir()
    for folder in ['results','models','data_audit','logs','executed_notebooks','cache/ipython']:(SECOND/folder).mkdir(parents=True,exist_ok=True)
    shutil.copytree(BASE/'src',SECOND/'src')
    shutil.copytree(BASE/'sources',SECOND/'sources')
    for name in ['requirements.txt','run_reproduction.py','predict.py']:shutil.copy2(BASE/name,SECOND/name)
    common=SECOND/'src/common.py';old=common.read_text()
    new=old.replace('ROOT = Path(__file__).resolve().parents[2]',"ROOT = Path(os.environ.get('REPRO_WORKSPACE_ROOT', Path(__file__).resolve().parents[2]))")
    new=new.replace("REP = ROOT / 'reproduction'","REP = Path(os.environ.get('REPRO_OUTPUT_ROOT', ROOT / 'reproduction'))")
    common.write_text(new)
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='run1/src/common.py',tofile='run2/src/common.py'))
    (SECOND/'logs/path_isolation_only.diff').write_text(patch)
    source_checks=[]
    for p in sorted((BASE/'src').glob('*.py')):
        q=SECOND/'src'/p.name
        source_checks.append(dict(file=p.name,run1_sha256=digest(p),run2_sha256=digest(q),equal=digest(p)==digest(q),reason='two path-routing lines only' if p.name=='common.py' else 'unchanged'))
    (SECOND/'logs/source_equivalence.json').write_text(json.dumps(source_checks,indent=2)+'\n')
    plan=dict(run1=str(BASE),run2=str(SECOND),source_data='same unchanged author repository',
        rerun_esm_inference=True,reuse_pretrained_weight_cache=True,skip_models=False,
        grid_split_seed=2001,eval_split_seeds=list(range(10)),cleanlab_cv_seed=0,
        scientific_hyperparameters_changed=False,rf_random_state='None, preserved from author source',
        lstm_weight_seeds='0 for split42; 0..9 for independent splits',
        path_changes=patch,baseline_files=len(manifest))
    (SECOND/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)

def execute():
    if not (SECOND/'run_plan.json').exists():raise RuntimeError('Prepare the isolated run directory before execution')
    if (SECOND/'run_status.json').exists():raise RuntimeError('Recorded run already exists; use a new output directory for another experiment')
    env=os.environ.copy();env.update(REPRO_WORKSPACE_ROOT=str(ROOT),REPRO_OUTPUT_ROOT=str(SECOND),
        TORCH_HOME=str(REP/'cache/torch'),HF_HOME=str(REP/'cache/huggingface'),
        MPLCONFIGDIR=str(SECOND/'cache/matplotlib'),KERAS_HOME=str(SECOND/'cache/keras'),
        PYTHONDONTWRITEBYTECODE='1')
    (SECOND/'logs/pip_freeze_before_run.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--disable-pip-version-check'],env=env,text=True))
    start=datetime.datetime.now(datetime.timezone.utc)
    with (SECOND/'logs/full_pipeline.log').open('w') as log:
        process=subprocess.run([sys.executable,str(SECOND/'run_reproduction.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(start_utc=start.isoformat(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=process.returncode)
    (SECOND/'run_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    raise SystemExit(process.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else execute()
