"""Predeclared sensitivity checks, never choose a run to match 339."""
from common import *
def run():
    import numpy as np,pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold,cross_val_predict
    from cleanlab.filter import find_label_issues
    raw,author,_=load_data();X=np.load(RESULT/'esm2_embeddings_recomputed.npy');y=raw.activity.to_numpy(dtype=int)
    records=[]
    for folds,seed,max_iter in [(5,s,5000) for s in range(10)]+[(10,0,5000),(5,0,100)]:
        probs=cross_val_predict(LogisticRegression(max_iter=max_iter),X,y,cv=StratifiedKFold(folds,shuffle=True,random_state=seed),method='predict_proba',n_jobs=2)
        bad=find_label_issues(y,probs,n_jobs=1)
        records.append(dict(cv_folds=folds,cv_seed=seed,max_iter=max_iter,retained=int((~bad).sum()),positive=int((~bad & (y==1)).sum()),negative=int((~bad & (y==0)).sum()),agreement_with_author=float((raw.No.isin(author.index)==~bad).mean()),design='independent sensitivity, not author algorithm'))
    pd.DataFrame(records).to_csv(RESULT/'cleanlab_sensitivity.csv',index=False)
if __name__=='__main__':run()
