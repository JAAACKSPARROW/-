from common import *
def run():
    import numpy as np,pandas as pd
    from rapidfuzz.distance import Levenshtein
    from sklearn.model_selection import train_test_split
    raw,X,y=load_data();mapped=raw.set_index('No').loc[X.index].reset_index()
    experiments=[('author_cleaned',mapped)]
    mask=RESULT/'cleanlab_keep_mask.npy'
    if mask.exists():experiments.append(('independent_cleaned',raw[np.load(mask)].reset_index(drop=True)))
    pairs=[];summary=[];splits=[]
    for name,df in experiments:
        for seed in [2001,*range(10)]:
            tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=seed)
            sims=np.array([[Levenshtein.normalized_similarity(df.iloc[i].Sequence,df.iloc[j].Sequence) for j in te] for i in tr])
            for i,j in zip(*np.where(sims>.8)):
                a=df.iloc[tr[i]];b=df.iloc[te[j]]
                pairs.append(dict(dataset=name,seed=seed,train_No=a.No,test_No=b.No,train_sequence=a.Sequence,test_sequence=b.Sequence,similarity=sims[i,j],train_label=a.activity,test_label=b.activity))
            summary.append(dict(dataset=name,seed=seed,train_size=len(tr),test_size=len(te),exact_duplicate_pairs=int((sims==1).sum()),pairs_gt_08=int((sims>.8).sum()),pairs_gt_09=int((sims>.9).sum()),test_sequences_gt_08=int((sims.max(0)>.8).sum()),test_sequences_gt_09=int((sims.max(0)>.9).sum()),maximum_similarity=float(sims.max())))
            for kind,ids in [('train',tr),('test',te)]:
                for i in ids:splits.append(dict(dataset=name,seed=seed,partition=kind,No=df.iloc[i].No))
    pd.DataFrame(pairs,columns=['dataset','seed','train_No','test_No','train_sequence','test_sequence','similarity','train_label','test_label']).to_csv(AUDIT/'train_test_similarity.csv',index=False)
    pd.DataFrame(summary).to_csv(AUDIT/'similarity_summary.csv',index=False)
    pd.DataFrame(splits).to_csv(AUDIT/'split_membership.csv',index=False)
    # Tune once on split 2001; its training set overlaps later evaluation test sets.
    tuning_tr,_=train_test_split(np.arange(len(mapped)),test_size=.2,random_state=2001)
    overlap=[]
    for seed in range(10):
        _,te=train_test_split(np.arange(len(mapped)),test_size=.2,random_state=seed)
        overlap.append(dict(seed=seed,test_size=len(te),test_rows_seen_in_hyperparameter_selection=len(set(te)&set(tuning_tr))))
    pd.DataFrame(overlap).to_csv(AUDIT/'hyperparameter_selection_overlap.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':run()
