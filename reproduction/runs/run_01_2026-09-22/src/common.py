"""Shared paths, immutable inputs and independently verified metrics."""
from pathlib import Path
import os, json, time, sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / 'reproduction'
AUTHOR = ROOT / 'skinaging_predictor_ZJU'
RESULT = REP / 'results'
AUDIT = REP / 'data_audit'
os.environ.setdefault('TORCH_HOME', str(REP / 'cache/torch'))
os.environ.setdefault('HF_HOME', str(REP / 'cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(REP / 'cache/matplotlib'))
os.environ.setdefault('KERAS_HOME', str(REP / 'cache/keras'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
# macOS LightGBM wheel expects libomp; reuse the installed PyTorch runtime.
# No system library or author source is modified.
if sys_platform := (os.uname().sysname == 'Darwin'):
    import sys
    omp = Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/torch/lib'
    if (omp/'libomp.dylib').exists():
        os.environ['DYLD_LIBRARY_PATH']=str(omp)+(':'+os.environ['DYLD_LIBRARY_PATH'] if os.environ.get('DYLD_LIBRARY_PATH') else '')
def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str)+'\n')
def load_data():
    import pandas as pd, numpy as np
    raw = pd.read_excel(AUTHOR/'Database/Oringinal_data.xlsx')
    raw = raw.dropna(how='all').reset_index(drop=True)
    X = pd.read_csv(AUTHOR/'Database/X_320_clean_LR.csv', index_col=0)
    y = np.loadtxt(AUTHOR/'Database/y_after_clean.csv', delimiter=',').astype(int)
    return raw, X, y
def metrics(y, pred, score):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0),
        specificity=tn/(tn+fp), f1=f1_score(y,pred,zero_division=0), mcc=matthews_corrcoef(y,pred),
        roc_auc=roc_auc_score(y,score), pr_auc=average_precision_score(y,score),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        author_printed_recall=tn/(tn+fn),author_printed_precision=tn/(tn+fp),
        author_printed_balanced_accuracy=.5*tn/(tn+fn)+.5*tp/(tp+fp),
        author_printed_f1=2*tn/(2*tn+fn+fp))
def evaluate(estimator,X,y,experiment,model,seeds=range(10)):
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    rows=[]
    for seed in seeds:
        tr,te=train_test_split(range(len(y)),test_size=.2,random_state=seed)
        clf=clone(estimator); start=time.perf_counter();clf.fit(X[tr],y[tr]);fit=time.perf_counter()-start
        start=time.perf_counter();pred=clf.predict(X[te])
        score=clf.decision_function(X[te]) if hasattr(clf,'decision_function') else clf.predict_proba(X[te])[:,1]
        row=dict(experiment=experiment,model=model,seed=seed,train_size=len(tr),test_size=len(te),
            training_seconds=fit,inference_seconds=time.perf_counter()-start,preprocessing='none',
            parameters=json.dumps(clf.get_params(),default=str,sort_keys=True),**metrics(y[te],pred,score))
        rows.append(row)
    return rows
