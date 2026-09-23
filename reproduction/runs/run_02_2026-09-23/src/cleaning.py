"""Independent CleanLab reconstruction: author's cleaning source/version absent."""
from common import *
def run(X):
    import numpy as np,pandas as pd,cleanlab
    from sklearn.model_selection import StratifiedKFold,cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    raw,author,ay=load_data();y=raw.activity.to_numpy(dtype=int)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    clf=LogisticRegression(max_iter=5000)
    probs=cross_val_predict(clf,X,y,cv=cv,method='predict_proba',n_jobs=2)
    bad=find_label_issues(y,probs,filter_by='prune_by_noise_rate',n_jobs=2)
    quality=get_label_quality_scores(y,probs,method='self_confidence')
    df=raw.rename(columns={'Sequence':'sequence','activity':'original_label'}).copy()
    df['predicted_probability']=probs[:,1];df['given_label_probability']=probs[np.arange(len(y)),y]
    df['issue_score']=quality;df['removed_or_kept']=np.where(bad,'removed','kept')
    df['author_kept']=df.No.isin(author.index);df['reproduction_kept']=~bad
    df.to_csv(RESULT/'cleanlab_all_samples.csv',index=False)
    df[bad].to_csv(RESULT/'cleanlab_removed_samples.csv',index=False)
    both=df.author_kept & ~bad
    summary=dict(status='INDEPENDENT_RECONSTRUCTION_NOT_EXACT_AUTHOR_REPRODUCTION',version=cleanlab.__version__,
        author_source='not publicly provided',base_classifier=str(clf),cv='StratifiedKFold(5, shuffle=True, random_state=0)',
        api='cross_val_predict + find_label_issues(filter_by=prune_by_noise_rate)',
        issue_score_definition='self_confidence: lower is more suspicious',threshold='CleanLab default; no manual tuning',
        intersection=int(both.sum()),author_only=int((df.author_kept & bad).sum()),
        reproduction_only=int((~df.author_kept & ~bad).sum()),agreement_rate=float((df.author_kept==~bad).mean()),
        author_retained=len(author),reproduction_retained=int((~bad).sum()),
        reproduction_positive=int(((~bad)&(y==1)).sum()),reproduction_negative=int(((~bad)&(y==0)).sum()))
    pd.DataFrame([summary]).to_csv(RESULT/'cleanlab_comparison.csv',index=False)
    write_json(RESULT/'cleanlab_configuration.json',summary)
    np.save(RESULT/'cleanlab_keep_mask.npy',~bad)
    print(summary,flush=True);return ~bad
if __name__=='__main__':
    import numpy as np
    run(np.load(RESULT/'esm2_embeddings_recomputed.npy'))
