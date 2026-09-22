"""Independent experiment with fixed Table S2 hyperparameters, not retuned."""
from common import *
def run():
    import pandas as pd,numpy as np,ctypes,sys
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    if os.uname().sysname=='Darwin':
        ctypes.CDLL(str(Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/torch/lib/libomp.dylib'))
    from lightgbm import LGBMClassifier
    _,df,y=load_data();X=df.to_numpy()
    models=dict(LR=LogisticRegression(C=1,class_weight='balanced',max_iter=5000,solver='saga',random_state=0),
        RF=RandomForestClassifier(max_depth=4,max_features='sqrt',n_estimators=160,random_state=0,n_jobs=2),
        KNN=KNeighborsClassifier(algorithm='auto',leaf_size=10,n_neighbors=3,weights='uniform',n_jobs=2),
        SVM=SVC(C=10,degree=1,kernel='poly',tol=1e-5),
        LightGBM=LGBMClassifier(boosting='gbdt',learning_rate=.5,n_estimators=40,objective='binary',reg_lambda=0,n_jobs=2,verbose=-1))
    for name,model in models.items():
        rows=evaluate(model,X,y,'F_TableS2_fixed_parameters',name)
        pd.DataFrame(rows).to_csv(RESULT/f'metrics_{name}_paper_fixed.csv',index=False)
        print('DONE TableS2',name,flush=True)
if __name__=='__main__':run()
