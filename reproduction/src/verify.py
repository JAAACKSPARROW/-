"""Independent final integrity and outcome checks (no retraining search)."""
from common import *
import hashlib,subprocess
def run():
    import numpy as np,pandas as pd,joblib
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    checks=[]
    for line in (AUDIT/'checksums.txt').read_text().splitlines():
        digest,rel=line.split('  ',1)
        assert hashlib.sha256((AUTHOR/rel).read_bytes()).hexdigest()==digest, rel
    checks.append('All inventoried original author file SHA256 unchanged')
    raw,X,y=load_data();r=np.load(RESULT/'esm2_embeddings_recomputed.npy')
    assert r.shape==(420,320) and X.shape==(339,320) and np.isfinite(r).all()
    full=pd.read_csv(RESULT/'original_notebook_outputs/output_name.csv',index_col=0).to_numpy()
    assert full.shape==r.shape
    d=np.abs(full-r);assert d.max()<2e-5
    write_json(RESULT/'esm_fullbatch_vs_minibatch.json',dict(max_abs_difference=float(d.max()),mean_abs_difference=float(d.mean()),shape=full.shape,source='actual executed path-only repaired author ESM notebook vs batch32 reconstruction'))
    checks.append('Author full-batch and reconstruction minibatch agree within 2e-5')
    m=pd.read_csv(RESULT/'model_metrics.csv')
    assert set(['LR','RF','KNN','SVM','LightGBM','LSTM'])<=set(m.model)
    for exp,df in m[m.model=='SVM'].groupby('experiment'):
        assert set(df.seed)==set(range(10)),exp
    assert len(m[(m.model=='LSTM')&(m.experiment=='E_independent_LSTM_10_splits')])==10
    assert np.isfinite(m[['accuracy','balanced_accuracy','precision','recall','specificity','f1','mcc','roc_auc','pr_auc','training_seconds','inference_seconds']]).all().all()
    assert ((m.tn+m.fp+m.fn+m.tp)==m.test_size).all()
    assert np.allclose(m.balanced_accuracy,.5*(m.tp/(m.tp+m.fn)+m.tn/(m.tn+m.fp)))
    checks.append('All six models have finite metrics/timing; repeated SVM experiments have seeds 0–9; confusion-matrix counts reconcile')
    # Refit only one SVM split to validate persisted metrics independently.
    clf=joblib.load(REP/'models/svm_final.joblib');tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=0)
    local=clone(clf).fit(X.to_numpy()[tr],y[tr]);p=local.predict(X.to_numpy()[te])
    saved=m[(m.experiment=='A_author_cleaned_code_grid')&(m.model=='SVM')&(m.seed==0)].iloc[0]
    calculated=metrics(y[te],p,local.decision_function(X.to_numpy()[te]))
    assert np.isclose(calculated['mcc'],saved.mcc) and np.isclose(calculated['balanced_accuracy'],saved.balanced_accuracy)
    checks.append('Independent seed0 SVM refit reproduces persisted MCC and standard BACC')
    for name,n in [('SVM',240),('LR',27),('RF',70),('KNN',896),('LightGBM',30)]:
        cv=pd.read_csv(RESULT/f'grid_{name}_code.csv');assert len(cv)==n,(name,len(cv))
        assert cv['mean_test_score'].notna().all(),name
    checks.append('All full original model grids completed with finite CV scores')
    required=['REPRODUCTION_REPORT.md','DATA_AUDIT.md','results/model_metrics.csv','results/paper_vs_reproduction.csv','requirements.txt','run_reproduction.py','predict.py','models/svm_final.joblib']
    for name in required:assert (REP/name).is_file() and (REP/name).stat().st_size>0,name
    (REP/'logs/local_file_verification.txt').write_text('\n'.join(str((REP/name).resolve()) for name in required)+'\n')
    write_json(RESULT/'verification.json',dict(status='PASS',checks=checks,metric_rows=len(m),required_files=required))
    print('\n'.join(checks),flush=True)
if __name__=='__main__':run()
