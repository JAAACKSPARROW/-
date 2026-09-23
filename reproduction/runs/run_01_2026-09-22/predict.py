"""Predict with the all-data refit; decision scores are not probabilities."""
import argparse,sys,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from common import *
def main():
    import joblib
    from embeddings import embed
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args()
    s=a.sequence
    if not s or any(c not in 'ARNDCEQGHILKMFPSTWYV' for c in s):p.error('sequence must contain uppercase standard amino acids only; no whitespace')
    model=joblib.load(REP/'models/svm_final.joblib')
    with contextlib.redirect_stdout(sys.stderr):X=embed([s])
    cls=int(model.predict(X)[0]);score=float(model.decision_function(X)[0])
    print(json.dumps(dict(Sequence=s,Predicted_class=cls,Predicted_class_label='potential anti-skin-aging peptide' if cls else 'random comparator / negative',Decision_score=score,Score_type='decision_function score (NOT probability)'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
