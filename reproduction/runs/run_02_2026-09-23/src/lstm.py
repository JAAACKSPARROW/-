"""Attempt exact author LSTM cell, then only repair target rank if required."""
from common import *
import traceback,contextlib,io,sys
def run():
    import numpy as np,pandas as pd,tensorflow as tf,keras
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,roc_auc_score,matthews_corrcoef
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,Dropout
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    keras.utils.set_random_seed(0)
    raw,X,y=load_data();X=X.to_numpy()
    nb=json.loads((AUTHOR/'Model_development/model_development_320_LR_s0.ipynb').read_text())
    source=''.join(nb['cells'][11]['source'])
    # Instrumentation only; original fit/predict calls are preserved.
    source=source.replace('model.fit(X_train, y_train,', 'fit_start=__import__("time").perf_counter()\nmodel.fit(X_train, y_train,')
    source=source.replace('y_pred = model.predict(X_test)', 'fit_seconds=__import__("time").perf_counter()-fit_start\ninfer_start=__import__("time").perf_counter()\ny_pred = model.predict(X_test)\ninfer_seconds=__import__("time").perf_counter()-infer_start')
    env=dict(X=X,y=y,Dropout=Dropout)
    try:
        exec(source,env);status='PASS';repair='none'
    except Exception:
        (REP/'logs/lstm_original_error.log').write_text(traceback.format_exc())
        # Only shape repair: sequence length is one, output (n,1,1) vs labels (n,).
        keras.backend.clear_session();keras.utils.set_random_seed(0)
        source=source.replace('model.fit(X_train, y_train,','model.fit(X_train, y_train.reshape(-1,1,1),')
        (REP/'src/lstm_minimal_repair_source.py.txt').write_text(source)
        exec(source,env);status='PASS_WITH_TARGET_RANK_REPAIR';repair='reshape targets (n,) -> (n,1,1) for Keras 3'
    row=dict(experiment='D_author_LSTM_split42_harness_seed0',model='LSTM',seed=42,train_size=len(env['y_train']),test_size=len(env['y_test']),preprocessing='reshape (n,320) -> (n,1,320)',training_seconds=env['fit_seconds'],inference_seconds=env['infer_seconds'],parameters='LSTM128/64/32; Dense64/10/1; Dropout .15; Adam; 100 epochs; batch32; harness seed0',**metrics(env['y_test'],env['y_pred_binary'],env['y_pred_proba']))
    pd.DataFrame([row]).to_csv(RESULT/'metrics_LSTM_code.csv',index=False)
    env['model'].save(REP/'models/lstm.keras')
    write_json(RESULT/'lstm_status.json',dict(status=status,repair=repair,author_weight_seed='not specified',harness_weight_seed=0,split_seed=42))
    rows=[]
    for seed in range(10):
        keras.backend.clear_session();keras.utils.set_random_seed(seed)
        repeated=source.replace('random_state=42',f'random_state={seed}').replace('verbose=1','verbose=0')
        env=dict(X=X,y=y,Dropout=Dropout)
        exec(repeated,env)
        rows.append(dict(experiment='E_independent_LSTM_10_splits',model='LSTM',seed=seed,train_size=len(env['y_train']),test_size=len(env['y_test']),preprocessing=row['preprocessing'],training_seconds=env['fit_seconds'],inference_seconds=env['infer_seconds'],parameters=row['parameters'].replace('harness seed0',f'harness seed{seed}'),**metrics(env['y_test'],env['y_pred_binary'],env['y_pred_proba'])))
        pd.DataFrame(rows).to_csv(RESULT/'metrics_LSTM_repeated.csv',index=False)
        print('DONE independent LSTM seed',seed,flush=True)
if __name__=='__main__':run()
