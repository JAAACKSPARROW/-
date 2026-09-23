"""Read-only audit. Preserve raw sequences and labels exactly."""
import sys, json, hashlib, csv, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def run():
    import pandas as pd, numpy as np
    rows=[]; checks=[];tree=[]
    for p in sorted(AUTHOR.rglob('*')):
        rel=p.relative_to(AUTHOR)
        if '.git' in rel.parts:continue
        tree.append('  '*(len(rel.parts)-1)+p.name+('/' if p.is_dir() else ''))
        if not p.is_file():continue
        ext=p.suffix.lower();cat='other';role='unclassified'
        if ext=='.xlsx':cat='raw peptide data';role='raw sequences and labels'
        elif p.name.startswith('X_'):cat='ESM embedding';role='cleaned 320D features with original No index'
        elif p.name.startswith('y_'):cat='cleaned data';role='cleaned labels, positional alignment with X'
        elif ext=='.ipynb':cat='notebooks';role='author code and stored outputs'
        elif ext=='.py':cat='Python scripts';role='evaluation helper'
        elif ext=='.md':cat='documentation';role='repository README'
        elif 'Sequence of collagens' in str(rel):cat='supplementary data';role='collagen sequences; not training peptides'
        rows.append(dict(path=str(rel),filename=p.name,extension=ext,size_bytes=p.stat().st_size,category=cat,likely_role=role))
        checks.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}')
    pd.DataFrame(rows).to_csv(REP/'repository_inventory.csv',index=False)
    (REP/'repository_tree.txt').write_text('\n'.join(tree)+'\n')
    (AUDIT/'checksums.txt').write_text('\n'.join(checks)+'\n')
    raw,X,y=load_data();raw.to_csv(AUDIT/'raw_data_readonly_export.csv',index=False)
    assert raw.No.is_unique and X.index.is_unique
    mapped=raw.set_index('No').loc[X.index].reset_index()
    assert np.array_equal(mapped.activity.to_numpy(),y), 'Author No-to-label alignment fails'
    mapped.to_csv(AUDIT/'author_cleaned_sequence_mapping.csv',index=False)
    summaries=[];issues=[]
    for name,df in [('Database/Oringinal_data.xlsx',raw),('Database/X_320_clean_LR.csv + y_after_clean.csv',mapped)]:
        s=df.Sequence;lengths=s.str.len();labels=df.activity
        conflicts=df.groupby('Sequence').activity.nunique();conflicts=conflicts[conflicts>1]
        summary=dict(path=name,format='xlsx' if name.endswith('xlsx') else 'csv pair',total=len(df),
            columns=json.dumps(df.columns.tolist()),label_definition='activity 1=ASAP, 0=random comparator',
            positive=int((labels==1).sum()),negative=int((labels==0).sum()),missing_values=int(df.isna().sum().sum()),
            duplicated_sequences=int(s.duplicated().sum()),duplicated_rows=int(df.duplicated().sum()),
            duplicated_sequence_label_rows=int(df[['Sequence','activity']].duplicated().sum()),
            min_length=int(lengths.min()),max_length=int(lengths.max()),mean_length=lengths.mean(),median_length=lengths.median(),
            conflicting_sequences=len(conflicts),abnormal_labels=int((~labels.isin([0,1])).sum()),
            invalid_sequences=int((~s.str.fullmatch('[ARNDCEQGHILKMFPSTWYV]+',na=False)).sum()))
        summaries.append(summary)
        lengths.value_counts().sort_index().rename_axis('length').reset_index(name='count').to_csv(AUDIT/('raw_length_distribution.csv' if name.endswith('xlsx') else 'cleaned_length_distribution.csv'),index=False)
        for _,r in df.iterrows():
            seq=r.Sequence;bad=not isinstance(seq,str) or not seq or any(c not in 'ARNDCEQGHILKMFPSTWYV' for c in seq)
            if bad or seq in conflicts.index or s.duplicated(keep=False).loc[r.name]:
                issues.append(dict(file=name,No=r.No,sequence=seq,label=r.activity,invalid=bad,
                    whitespace=any(c.isspace() for c in seq) if isinstance(seq,str) else False,
                    lowercase=any(c.islower() for c in seq) if isinstance(seq,str) else False,
                    conflicting_label=seq in conflicts.index,duplicate=bool(s.duplicated(keep=False).loc[r.name])))
    # Separate file-level facts for both CSVs, avoiding invented sequence columns.
    summaries.append(dict(path='Database/X_320_clean_LR.csv',format='csv',total=len(X),columns=json.dumps(X.columns.tolist()),
        missing_values=int(X.isna().sum().sum()),duplicated_rows=int(X.duplicated().sum()),label_definition='labels in separate y file; sequence mapped using No'))
    summaries.append(dict(path='Database/y_after_clean.csv',format='csv headerless',total=len(y),columns='[label]',
        positive=int((y==1).sum()),negative=int((y==0).sum()),missing_values=int(np.isnan(y).sum()),
        duplicated_rows=len(y)-len(np.unique(y)),label_definition='1=ASAP, 0=random; repeats expected for labels'))
    pd.DataFrame(summaries).to_csv(AUDIT/'data_summary.csv',index=False)
    pd.DataFrame(issues,columns=['file','No','sequence','label','invalid','whitespace','lowercase','conflicting_label','duplicate']).to_csv(AUDIT/'sequence_issues.csv',index=False)
    write_json(AUDIT/'embedding_file_info.json',dict(shape=X.shape,dtype=str(X.to_numpy().dtype),
        original_indices=X.index.tolist(),mapping='No joins raw No, label agreement verified',finite=bool(np.isfinite(X).all().all())))
    text='# Data audit\n\nAll author files are read-only. checksums.txt covers every non-Git author file.\n\n'
    for row in summaries[:2]:text+=f"- {row['path']}: n={row['total']}, positive={row['positive']}, negative={row['negative']}, sequence duplicates={row['duplicated_sequences']}, conflicts={row['conflicting_sequences']}, invalid={row['invalid_sequences']}, lengths={row['min_length']}–{row['max_length']}.\n"
    text+='\nPaper claims raw 420 (210/210), cleaned 339 (161/178). See calculated CSV; blank trailing Excel formatting rows are not observations. Raw duplicate full rows include the unique No column; sequence/label duplicate counts are reported separately.\n\nCleaned CSV No indices join to raw No with exact label agreement; the current cleaned CSV contains 339 rows, whereas stored training notebook outputs show 343 rows and different feature values. Thus saved notebook outputs are not results of the current published input files.\n'
    (AUDIT/'DATA_AUDIT.md').write_text(text);(REP/'DATA_AUDIT.md').write_text(text)
    print(pd.DataFrame(summaries[:2]).to_string(index=False),flush=True)
    return raw,X,y
if __name__=='__main__':run()
