"""Execute author grid-search source and independent standard-metric evaluation."""
from common import *
import sys,ast,traceback,contextlib,io,copy
sys.path.insert(0,str(AUTHOR))
CELL_MODEL={4:'LR',5:'RF',6:'KNN',7:'SVM',8:'LightGBM'}
def source_grid(cell):
    # Exact source up through grid_search.best_estimator_. Drop inaccessible screening section.
    s=''.join(cell['source']);return s[:s.index('best_model_reg = grid_search.best_estimator_')]+ 'best_model_reg = grid_search.best_estimator_\n'
def grids():
    nb=json.loads((AUTHOR/'Model_development/model_development_320_LR_s0.ipynb').read_text())
    return {name:source_grid(nb['cells'][i]) for i,name in CELL_MODEL.items()}
def fit_grid(source,X,y,scoring=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import make_scorer,matthews_corrcoef
    from threadpoolctl import threadpool_limits
    from joblib import parallel_backend
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=2001)
    env=dict(X_train_whole=xt,y_train_whole=yt)
    if 'import lightgbm' in source and os.uname().sysname=='Darwin':
        import ctypes
        ctypes.CDLL(str(Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/torch/lib/libomp.dylib'))
    if scoring=='mcc':
        source=source.replace('cv=10,n_jobs=-1','cv=10,n_jobs=-1,scoring=make_scorer(matthews_corrcoef)')
        env.update(make_scorer=make_scorer,matthews_corrcoef=matthews_corrcoef)
    # Same grid/order/score; deterministic supplemental seeds only for otherwise unseeded RF.
    start=time.perf_counter()
    with threadpool_limits(limits=1),parallel_backend('loky',n_jobs=2):exec(source,env)
    return env['grid_search'],time.perf_counter()-start
def run(models=None):
    import numpy as np,pandas as pd,joblib,nbformat
    from sklearn.base import clone
    from self_function import evaluation
    raw,df,y=load_data();X=df.to_numpy();sources=grids()
    selected=models or list(CELL_MODEL.values());allrows=[]
    status_path=RESULT/'model_run_status.csv'
    statuses=pd.read_csv(status_path).to_dict('records') if status_path.exists() else []
    for name in selected:
        print('START author grid',name,flush=True)
        np.random.seed(0) # added harness seed; author's stochastic estimator seeds are absent
        t=time.perf_counter()
        try:
            search,elapsed=fit_grid(sources[name],X,y)
            best=search.best_estimator_
            pd.DataFrame(search.cv_results_).to_csv(RESULT/f'grid_{name}_code.csv',index=False)
            write_json(RESULT/f'grid_{name}_code_summary.json',dict(parameters=search.best_params_,best_cv_score=search.best_score_,scoring='accuracy (default)',cv=10,split_seed=2001,search_seconds=elapsed,train_size=271,test_size=68,scaler=None,harness_numpy_seed=0,estimator_random_state=best.get_params().get('random_state','not applicable')))
            out=io.StringIO()
            with contextlib.redirect_stdout(out):evaluation(clone(best),X,y)
            (REP/f'logs/author_evaluation_{name}.txt').write_text(out.getvalue())
            rows=evaluate(best,X,y,'A_author_cleaned_code_grid',name)
            pd.DataFrame(rows).to_csv(RESULT/f'metrics_{name}_code.csv',index=False);allrows+=rows
            joblib.dump(best,REP/f'models/{name}_author_grid.joblib')
            # A reproducible executed notebook containing the exact author grid fragment.
            nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Extracted author grid executed by exec; input paths resolved by harness. Complete original attempts saved separately. Evaluation below uses original helper, including its metric naming bug.'),nbformat.v4.new_code_cell(sources[name],execution_count=1,outputs=[nbformat.v4.new_output('stream',name='stdout',text=out.getvalue()+str(search.best_params_))])])
            nbformat.write(nb,REP/f'executed_notebooks/extracted_{name}_grid.ipynb')
            statuses.append(dict(model=name,status='PASS',seconds=time.perf_counter()-t))
            print('DONE',name,search.best_params_,flush=True)
        except Exception as e:
            error=traceback.format_exc();(REP/f'logs/model_{name}_error.log').write_text(error)
            statuses.append(dict(model=name,status='FAIL',error=str(e),seconds=time.perf_counter()-t));print(error,flush=True)
        pd.DataFrame(statuses).to_csv(RESULT/'model_run_status.csv',index=False)
    return allrows
def svm_experiments():
    import numpy as np,pandas as pd,joblib
    from sklearn.base import clone
    raw,df,y=load_data();source=grids()['SVM'];X=df.to_numpy()
    search,elapsed=fit_grid(source,X,y,'mcc');best=search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(RESULT/'grid_SVM_mcc.csv',index=False)
    write_json(RESULT/'grid_SVM_mcc_summary.json',dict(parameters=search.best_params_,best_cv_score=search.best_score_,scoring='MCC',search_seconds=elapsed))
    rows=evaluate(best,X,y,'B_author_cleaned_MCC_grid','SVM')
    pd.DataFrame(rows).to_csv(RESULT/'metrics_SVM_mcc.csv',index=False)
    paths=[RESULT/'metrics_SVM_code.csv',RESULT/'metrics_SVM_mcc.csv']
    pd.concat([pd.read_csv(p) for p in paths]).to_csv(RESULT/'svm_code_vs_paper.csv',index=False)
    recomputed=np.load(RESULT/'esm2_embeddings_recomputed.npy');keep=np.load(RESULT/'cleanlab_keep_mask.npy')
    rx=recomputed[keep];ry=raw.activity.to_numpy(dtype=int)[keep]
    # Reconstruction follows code's grid scoring; not optimized to paper target.
    s,elapsed=fit_grid(source,rx,ry)
    pd.DataFrame(s.cv_results_).to_csv(RESULT/'grid_SVM_reconstructed.csv',index=False)
    r=evaluate(s.best_estimator_,rx,ry,'C_independent_ESM_CleanLab_code_grid','SVM')
    pd.DataFrame(r).to_csv(RESULT/'metrics_SVM_reconstructed.csv',index=False)
    write_json(RESULT/'grid_SVM_reconstructed_summary.json',dict(parameters=s.best_params_,best_cv_score=s.best_score_,search_seconds=elapsed,scoring='accuracy',cleanlab='independent reconstruction'))
    # Prediction artifact: refit code-selected SVM on all published cleaned samples.
    a=joblib.load(REP/'models/SVM_author_grid.joblib');final=clone(a).fit(X,y)
    joblib.dump(final,REP/'models/svm_final.joblib')
    write_json(REP/'models/configuration.json',dict(training_data='author published cleaned 339 rows',
        artifact_role='refit all cleaned data for prediction; not used for reported held-out metrics',
        parameters=final.get_params(),checkpoint='esm2_t6_8M_UR50D',layer=6,pooling='residue mean excluding BOS/EOS',
        scaler=None,label_mapping={'0':'random comparator / predicted negative','1':'potential anti-skin-aging peptide'},
        output_score='decision_function score, not probability',feature_dimension=320))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='*');p.add_argument('--svm-experiments',action='store_true');a=p.parse_args()
    if a.svm_experiments:svm_experiments()
    else:run(a.models)
