"""Author ESM-2 layer/pooling, independently regenerated from all raw sequences."""
from common import *
def embed(sequences, batch_size=32):
    import torch, esm, numpy as np
    torch.set_num_threads(2)
    model,alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    model.eval();converter=alphabet.get_batch_converter();out=[]
    for start in range(0,len(sequences),batch_size):
        seqs=sequences[start:start+batch_size]
        _,_,tokens=converter([(s,s) for s in seqs])
        lens=(tokens!=alphabet.padding_idx).sum(1)
        with torch.no_grad():r=model(tokens,repr_layers=[6],return_contacts=True)['representations'][6]
        out.extend(r[i,1:n-1].mean(0).numpy() for i,n in enumerate(lens))
        print(f'ESM {min(start+batch_size,len(sequences))}/{len(sequences)}',flush=True)
    return np.stack(out)
def run(skip=False):
    import numpy as np,pandas as pd,hashlib
    raw,author,y=load_data();path=RESULT/'esm2_embeddings_recomputed.npy';map_path=RESULT/'esm2_embedding_sequences.csv'
    if skip:
        X=np.load(path);old=pd.read_csv(map_path)
        assert old.Sequence.tolist()==raw.Sequence.tolist() and old.No.tolist()==raw.No.tolist(), 'cached embedding sequence order mismatch'
        assert X.shape==(len(raw),320) and np.isfinite(X).all()
    else:
        X=embed(raw.Sequence.tolist());np.save(path,X)
        raw.to_csv(map_path,index=False)
    rows=raw.set_index('No').index.get_indexer(author.index);ours=X[rows].astype(float);a=author.to_numpy()
    diff=ours-a;cos=(ours*a).sum(1)/(np.linalg.norm(ours,axis=1)*np.linalg.norm(a,axis=1))
    df=pd.DataFrame(dict(No=author.index,sequence=raw.iloc[rows].Sequence.to_numpy(),cosine_similarity=cos,
        MAE=np.abs(diff).mean(1),RMSE=np.sqrt((diff**2).mean(1)),maximum_absolute_difference=np.abs(diff).max(1)))
    df.to_csv(RESULT/'embedding_comparison.csv',index=False)
    summary=dict(shape=X.shape,dtype=str(X.dtype),mean_cosine=float(cos.mean()),median_cosine=float(np.median(cos)),
        minimum_cosine=float(cos.min()),mean_MAE=float(np.abs(diff).mean()),maximum_absolute_difference=float(np.abs(diff).max()),
        checkpoint='esm2_t6_8M_UR50D',layer=6,pooling='residue mean, exclude BOS/EOS/padding',device='cpu',batch_size=32,
        author_batch_size='all sequences in one batch',normalization='none',eval=True,no_grad=True,
        cache_sequence_order_verified=True)
    write_json(RESULT/'embedding_summary.json',summary)
    weights=[]
    for f in (REP/'cache/torch').rglob('*.pt'):weights.append(dict(path=str(f.relative_to(ROOT)),size_bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    write_json(RESULT/'esm_checkpoint_checksums.json',weights)
    print(summary,flush=True);return X
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--skip-esm',action='store_true');a=p.parse_args();run(skip=a.skip_esm)
