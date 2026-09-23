"""Compare independent complete executions without selecting seeds or changing labels."""
from pathlib import Path
import ast,hashlib,json,re,sys,datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reproduction';BASE=REP/'runs/run_01_2026-09-22';SECOND=REP/'runs/run_02_2026-09-23'
OUT=REP/'comparisons/run_01_vs_run_02'
METRICS=['accuracy','balanced_accuracy','precision','recall','specificity','f1','mcc','roc_auc','pr_auc']
ALIASES={'A_author_cleaned_code_grid':'A｜作者数据＋代码网格','B_author_cleaned_MCC_grid':'B｜作者数据＋MCC网格','C_independent_ESM_CleanLab_code_grid':'C｜重新编码＋独立清洗','D_author_LSTM_split42_harness_seed0':'D｜LSTM原划分42','E_independent_LSTM_10_splits':'E｜LSTM十次独立划分','F_TableS2_fixed_parameters':'F｜论文固定参数'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
def put(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')
def md(df):
    def val(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def author_references():
    raw=js(ROOT/'skinaging_predictor_ZJU/Model_development/model_development_320_LR_s0.ipynb')
    out=[];params=[]
    map_metrics={'BACC':'balanced_accuracy','Recall':'recall','Precision':'precision','MCC':'mcc','F1 score':'f1','ROC_AUC':'roc_auc'}
    for idx,name in {4:'LR',5:'RF',6:'KNN',7:'SVM',8:'LightGBM'}.items():
        text='\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in raw['cells'][idx].get('outputs',[]))
        for printed,key in map_metrics.items():
            match=re.search(re.escape(printed)+r'\s*=\s*([\d.]+)\s*±\s*([\d.]+)',text)
            if match:out.append(dict(source='author_stored_notebook',model=name,metric=key,mean=float(match[1]),std=float(match[2]),semantics='standard' if key in ['mcc','roc_auc'] else 'author_printed_label_not_standard_definition',source_cell=idx))
        match=re.search(r'Best Parameters:(\{[^\n]+\})',text)
        if match:params.append(dict(source='author_stored_notebook',model=name,parameters=json.dumps(ast.literal_eval(match[1]),sort_keys=True)))
    lstm='\n'.join(''.join(o.get('text',[])) for o in raw['cells'][11].get('outputs',[]))
    for label,key in [('MCC','mcc'),('ROC AUC','roc_auc')]:
        match=re.search(re.escape(label)+r':\s*([\d.]+)',lstm)
        if match:out.append(dict(source='author_stored_notebook',model='LSTM',metric=key,mean=float(match[1]),std=np.nan,semantics='single split42; standard definition',source_cell=11))
    # No invented BACC from rounded classification reports.
    tab=js(REP/'sources/supplementary_tables.json')[1]
    for row in tab[1:]:
        name={'LGBM':'LightGBM','RNN':'LSTM'}.get(row[0],row[0]);params.append(dict(source='paper_Table_S2',model=name,parameters=row[1]))
        for j,col in enumerate(tab[0][2:],start=2):
            mean,std=map(float,row[j].split('±'));out.append(dict(source='paper_Table_S2',model=name,metric=map_metrics[col],mean=mean,std=std,semantics='paper-reported; helper naming and CV-vs-repeat ambiguity',source_cell='Table S2'))
    return pd.DataFrame(out),pd.DataFrame(params)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert js(SECOND/'run_status.json')['exit_code']==0,'Second execution must finish successfully first'
    assert js(SECOND/'results/verification.json')['status']=='PASS'
    # Ensure archived run 1 and its original source artifacts have not changed.
    files=js(BASE/'archive_manifest.json')['files']
    assert all(sha(BASE/r['path'])==r['sha256'] for r in files)
    inputs=[]
    for line in (BASE/'data_audit/checksums.txt').read_text().splitlines():
        expected,path=line.split('  ',1);actual=sha(ROOT/'skinaging_predictor_ZJU'/path)
        inputs.append(dict(path=path,run1_sha256=expected,current_sha256=actual,identical=expected==actual))
    assert all(row['identical'] for row in inputs)
    pd.DataFrame(inputs).to_csv(OUT/'input_integrity.csv',index=False)
    a=pd.read_csv(BASE/'results/model_metrics.csv');b=pd.read_csv(SECOND/'results/model_metrics.csv')
    keys=['experiment','model','seed']
    joined=a.merge(b,on=keys,suffixes=('_run1','_run2'),how='outer',indicator=True,validate='one_to_one')
    assert (joined._merge=='both').all() and len(joined)==131
    joined=joined.drop(columns='_merge')
    for metric in METRICS:joined[f'{metric}_delta']=joined[f'{metric}_run2']-joined[f'{metric}_run1']
    joined.to_csv(OUT/'paired_seed_results.csv',index=False)
    summaries=[]
    for (experiment,model),d in joined.groupby(['experiment','model']):
        for metric in METRICS:
            x=d[f'{metric}_run1'];y=d[f'{metric}_run2'];delta=y-x
            summaries.append(dict(experiment=experiment,model=model,metric=metric,n=len(d),run1_mean=x.mean(),run1_std=x.std(ddof=1),run2_mean=y.mean(),run2_std=y.std(ddof=1),mean_delta=delta.mean(),max_abs_seed_delta=delta.abs().max(),changed_seeds=int((delta.abs()>1e-12).sum()),all_equal_at_1e_12=bool(np.allclose(x,y,rtol=0,atol=1e-12))))
    s=pd.DataFrame(summaries);s.to_csv(OUT/'model_repeatability_summary.csv',index=False)
    refs,params=author_references();refs.to_csv(OUT/'original_reference_metrics.csv',index=False)
    # Compare recorded author metrics only against the matching original helper expression.
    author_columns={'balanced_accuracy':'author_printed_balanced_accuracy','recall':'author_printed_recall','precision':'author_printed_precision','f1':'author_printed_f1','mcc':'mcc','roc_auc':'roc_auc'}
    authors=[]
    for _,ref in refs[refs.source=='author_stored_notebook'].iterrows():
        exp='D_author_LSTM_split42_harness_seed0' if ref.model=='LSTM' else 'A_author_cleaned_code_grid'
        col=author_columns.get(ref.metric,ref.metric)
        for name,frame in [('run1',a),('run2',b)]:
            d=frame[(frame.model==ref.model)&(frame.experiment==exp)]
            if len(d):authors.append(dict(run=name,model=ref.model,printed_metric=ref.metric,computed_column=col,author_stored_mean=ref['mean'],current_mean=d[col].mean(),signed_difference=d[col].mean()-ref['mean'],limitation='same formula, but stored notebook data/version differs; stochastic original weights/seeds unavailable'))
    pd.DataFrame(authors).to_csv(OUT/'author_same_formula_comparison.csv',index=False)
    paper=s.merge(refs[refs.source=='paper_Table_S2'][['model','metric','mean','std']],on=['model','metric'])
    paper=paper.rename(columns={'mean':'paper_mean','std':'paper_std'});paper['run2_minus_paper']=paper.run2_mean-paper.paper_mean
    paper['absolute_difference_from_paper']=paper.run2_minus_paper.abs();paper.to_csv(OUT/'paper_vs_both_runs.csv',index=False)
    parameter_rows=[]
    for model in ['LR','RF','KNN','SVM','LightGBM']:
        one=js(BASE/f'results/grid_{model}_code_summary.json');two=js(SECOND/f'results/grid_{model}_code_summary.json')
        parameter_rows.append(dict(model=model,run1_parameters=json.dumps(one['parameters'],sort_keys=True),run2_parameters=json.dumps(two['parameters'],sort_keys=True),parameters_equal=one['parameters']==two['parameters'],run1_cv_accuracy=one['best_cv_score'],run2_cv_accuracy=two['best_cv_score'],run1_grid_seconds=one['search_seconds'],run2_grid_seconds=two['search_seconds']))
        params=pd.concat([params,pd.DataFrame([dict(source='run1_code_grid',model=model,parameters=json.dumps(one['parameters'],sort_keys=True)),dict(source='run2_code_grid',model=model,parameters=json.dumps(two['parameters'],sort_keys=True))])],ignore_index=True)
    params.to_csv(OUT/'all_sources_parameters.csv',index=False);pd.DataFrame(parameter_rows).to_csv(OUT/'selected_parameters_comparison.csv',index=False)
    # Compare ALL numerical CV candidates, not only the selected winner.
    cv=[]
    for model in ['LR','RF','KNN','SVM','LightGBM']:
        d1=pd.read_csv(BASE/f'results/grid_{model}_code.csv');d2=pd.read_csv(SECOND/f'results/grid_{model}_code.csv')
        assert d1.params.tolist()==d2.params.tolist()
        delta=d2.mean_test_score-d1.mean_test_score
        for i,diff in enumerate(delta):cv.append(dict(model=model,candidate_index=i,parameters=d1.params.iloc[i],run1_mean_cv_score=d1.mean_test_score.iloc[i],run2_mean_cv_score=d2.mean_test_score.iloc[i],delta=diff))
    pd.DataFrame(cv).to_csv(OUT/'paired_grid_scores.csv',index=False)
    x1=np.load(BASE/'results/esm2_embeddings_recomputed.npy');x2=np.load(SECOND/'results/esm2_embeddings_recomputed.npy')
    seq1=pd.read_csv(BASE/'results/esm2_embedding_sequences.csv');seq2=pd.read_csv(SECOND/'results/esm2_embedding_sequences.csv')
    assert seq1.equals(seq2)
    error=np.abs(x2.astype(float)-x1.astype(float));c1=pd.read_csv(BASE/'results/cleanlab_all_samples.csv');c2=pd.read_csv(SECOND/'results/cleanlab_all_samples.csv')
    assert c1.No.tolist()==c2.No.tolist()
    k1=np.load(BASE/'results/cleanlab_keep_mask.npy');k2=np.load(SECOND/'results/cleanlab_keep_mask.npy')
    split_equal=(BASE/'data_audit/split_membership.csv').read_bytes()==(SECOND/'data_audit/split_membership.csv').read_bytes()
    q=pd.DataFrame(dict(No=seq1.No,sequence=seq1.Sequence,embedding_max_abs_delta=error.max(axis=1),embedding_mean_abs_delta=error.mean(axis=1),run1_kept=k1,run2_kept=k2,cleaning_agrees=k1==k2,probability_delta=c2.predicted_probability-c1.predicted_probability))
    q.to_csv(OUT/'embedding_cleanlab_paired_samples.csv',index=False)
    ev1=js(BASE/'logs/environment_versions.json');ev2=js(SECOND/'logs/environment_versions.json')
    pd.DataFrame([dict(component=k,run1=v,run2=ev2.get(k),equal=v==ev2.get(k)) for k,v in ev1.items()]).to_csv(OUT/'environment_comparison.csv',index=False)
    # Models fit on all 339 rows: compare output behavior, not misleading held-out accuracy.
    import joblib
    one=joblib.load(BASE/'models/svm_final.joblib');two=joblib.load(SECOND/'models/svm_final.joblib')
    p1=one.predict(x1);p2=two.predict(x1);v1=one.decision_function(x1);v2=two.decision_function(x1)
    pd.DataFrame(dict(No=seq1.No,sequence=seq1.Sequence,run1_class=p1,run2_class=p2,run1_score=v1,run2_score=v2,score_delta=v2-v1,classes_agree=p1==p2)).to_csv(OUT/'svm_prediction_agreement_420.csv',index=False)
    comparison=dict(paired_metric_rows=len(joined),archive_manifest_verified=len(files),all_author_files_unchanged=True,environment_equal=ev1==ev2,
        esm_bitwise_equal=bool(np.array_equal(x1,x2)),esm_max_abs_difference=float(error.max()),esm_mean_abs_difference=float(error.mean()),
        cleanlab_keep_masks_equal=bool(np.array_equal(k1,k2)),run1_cleanlab_kept=int(k1.sum()),run2_cleanlab_kept=int(k2.sum()),cleanlab_changed_decisions=int((k1!=k2).sum()),cleanlab_probability_max_abs_difference=float(np.abs(q.probability_delta).max()),
        split_membership_identical=split_equal,svm_all420_disagreements=int((p1!=p2).sum()),svm_all420_max_score_difference=float(np.abs(v2-v1).max()),
        svm_support_indices_equal=bool(np.array_equal(one.support_,two.support_)),svm_support_vectors_max_abs_delta=float(np.abs(one.support_vectors_-two.support_vectors_).max()) if one.support_vectors_.shape==two.support_vectors_.shape else None,
        all_selected_parameters_equal=all(r['parameters_equal'] for r in parameter_rows),
        changed_experiment_models=s.loc[~s.all_equal_at_1e_12,['experiment','model']].drop_duplicates().to_dict('records'),
        actual_run2_status=js(SECOND/'run_status.json'),publication_scope='Original Git history bundle excluded; models/data/logs allowed by prior user approval')
    put('comparison_verification.json',comparison)
    build_report(a,b,s,refs,params,pd.DataFrame(parameter_rows),comparison)
    print(json.dumps(comparison,indent=2,ensure_ascii=False),flush=True)

def build_report(a,b,s,refs,params,selected,check):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(REP/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def entry(experiment,model,metric):return s[(s.experiment==experiment)&(s.model==model)&(s.metric==metric)].iloc[0]
    def rounded(mean,std):return f'{mean:.6f} ± {std:.6f}' if pd.notna(std) else f'{mean:.6f}（单次）'
    rows=[]
    for name in ['LR','RF','KNN','SVM','LightGBM','LSTM']:
        exp='E_independent_LSTM_10_splits' if name=='LSTM' else 'A_author_cleaned_code_grid'
        ba=entry(exp,name,'balanced_accuracy');mc=entry(exp,name,'mcc')
        paper=refs[(refs.source=='paper_Table_S2')&(refs.model==name)&(refs.metric=='mcc')].iloc[0]
        stored=refs[(refs.source=='author_stored_notebook')&(refs.model==name)&(refs.metric=='mcc')].iloc[0]
        rows.append({'模型':name,'第一次 BACC':rounded(ba.run1_mean,ba.run1_std),'第二次 BACC':rounded(ba.run2_mean,ba.run2_std),
            '第一次 MCC':rounded(mc.run1_mean,mc.run1_std),'第二次 MCC':rounded(mc.run2_mean,mc.run2_std),'MCC 均值变化':mc.mean_delta,
            '作者保存 MCC':rounded(stored['mean'],stored['std']),'论文 MCC':rounded(paper['mean'],paper['std'])})
    main=pd.DataFrame(rows);main.to_csv(OUT/'all_models_readable_comparison.csv',index=False)
    stepfacts=[]
    for number,name,purpose,actual,issue in [
      (1,'固定输入与环境','保证第二次是可比较的重复执行','同一作者 commit、Python、依赖、数据；第一次170项产物快照校验通过','仅改两行输出路径；未改算法/网格/阈值'),
      (2,'审计原始数据','检查模型到底学习什么','420条：210阳性＋210随机对照；作者清洗后339条：161＋178','随机对照不等于实验验证无活性'),
      (3,'执行作者 Notebook','验证原始代码能否直接运行','原样失败；路径修复后ESM成功、训练Notebook在X_out处中断；保留错误','实际训练用原Notebook提取的完整网格和必要兼容修复'),
      (4,'重新计算 ESM-2','把不同长度序列转换为定长特征','esm2_t6_8M_UR50D，第6层，残基均值池化，420×320；本次重新前向计算','只复用预训练权重缓存，没有跳过embedding计算'),
      (5,'核对作者特征','确认编码和样本顺序','339条公开向量按No与原始序列连接；分别与两次重算向量比较','两次一致与作者一致是两种不同检验'),
      (6,'重新做 CleanLab','识别疑似标签问题','LR＋5折OOF概率＋默认过滤；两次保留'+str(check['run1_cleanlab_kept'])+'/'+str(check['run2_cleanlab_kept'])+'条','作者清洗实现缺失；353条是独立重建，不是严格复现339'),
      (7,'划分与网格搜索','在训练部分比较超参数','先seed2001做80/20划分；271条训练做10折网格，68条初始留出','原源码scoring默认accuracy；另做MCC评分作为B实验'),
      (8,'重复训练与评估','衡量对划分的敏感程度','固定选出的参数，用seeds0…9再训练/测试；标准差ddof=1','并非挑最好seed；后续测试集与此前调参数据有重叠'),
      (9,'复现其他模型','确认现象是否只存在于SVM','LR、RF、KNN、SVM、LightGBM完整网格；LSTM原split42＋十次独立划分','RF random_state=None原样保留；LSTM只作Keras标签维度兼容修复'),
      (10,'比较模型产物','判断两次预测行为是否改变','最终SVM分别在全部339条上refit，再对同一420条作一致性检查','这420条不是新的独立测试集，只用于诊断两模型差别'),
      (11,'对比论文和作者输出','分清数值差异与定义差异','同时保留论文S2、作者Notebook历史输出、两次重跑和论文固定参数实验','原权重未公开；保存输出至少展示343条，与当前339条不一致'),
      (12,'保存证据与解释','让每个结论可追溯','两次完整目录＋逐seed指标差＋参数差＋流程图＋哈希','稳定重复不消除原设计泄漏，也不证明生物学有效性')]:
        stepfacts.append(dict(步骤=number,名称=name,目的=purpose,本次执行=actual,理解要点=issue))
    pd.DataFrame(stepfacts).to_csv(OUT/'step_by_step_workflow.csv',index=False)
    flow='''flowchart TD
    S1["1 固定作者版本、环境、数据<br/>保留第一次快照"] --> S2["2 审计420条原始肽<br/>210阳性＋210随机对照"]
    S2 --> NB["3 尝试作者Notebook并记录错误<br/>执行必要修复副本"]
    NB --> S3["4 重算 ESM-2<br/>每条序列 → 320维向量"]
    S3 --> S4["5 按No核对作者339条特征<br/>同时对比两次embedding"]
    S4 --> A["作者提供的339条清洗数据"]
    S3 --> C["6 独立CleanLab清洗<br/>保留353条，原算法未公开"]
    A --> AB["7A 原代码完整网格<br/>accuracy评分；另做MCC评分"]
    A --> F["7B 论文Table S2固定参数<br/>不重新调参"]
    C --> CC["7C 用独立清洗数据<br/>训练SVM"]
    AB --> E["8 seeds 0…9训练/测试<br/>计算标准指标的均值±标准差"]
    F --> E
    CC --> E
    E --> O["9 汇总五种传统模型<br/>另跑LSTM原split42及十次划分"]
    O --> M["10 对比两次最终SVM预测<br/>仅检验一致性"]
    M --> R["11 对比第一次、第二次<br/>论文、作者历史输出"]
    R --> L["12 保存证据与流程解释<br/>注明指标定义及泄漏差异"]
'''
    training_flow='''flowchart LR
    D["作者清洗数据339条"] --> T["seed2001划分<br/>训练271／留出68"]
    T --> G["在271条上做10折CV<br/>选择一次超参数"]
    G --> P["固定这组参数"]
    P --> R["在全体339条上<br/>重新做seeds0…9划分"]
    R --> V["每次271条训练<br/>68条测试"]
    V --> O["10组指标→均值±样本SD"]
    G -. "后续测试样本可能参加过此前调参" .-> V
'''
    (OUT/'workflow.mmd').write_text(flow)
    (OUT/'training_and_leakage.mmd').write_text(training_flow)
    sv=[]
    for exp in ['A_author_cleaned_code_grid','B_author_cleaned_MCC_grid','C_independent_ESM_CleanLab_code_grid','F_TableS2_fixed_parameters']:
        ba=entry(exp,'SVM','balanced_accuracy');mc=entry(exp,'SVM','mcc');rc=entry(exp,'SVM','recall');pr=entry(exp,'SVM','precision')
        sv.append({'方案':ALIASES[exp],'第一次 BACC':rounded(ba.run1_mean,ba.run1_std),'第二次 BACC':rounded(ba.run2_mean,ba.run2_std),
            '第一次 MCC':rounded(mc.run1_mean,mc.run1_std),'第二次 MCC':rounded(mc.run2_mean,mc.run2_std),
            '第二次 Recall':rounded(rc.run2_mean,rc.run2_std),'第二次 Precision':rounded(pr.run2_mean,pr.run2_std)})
    svm=pd.DataFrame(sv);svm.to_csv(OUT/'svm_design_comparison.csv',index=False)
    changed=check['changed_experiment_models'];changes='；'.join(ALIASES[r['experiment']]+' / '+r['model'] for r in changed) or '无'
    rfc=selected[selected.model=='RF'].iloc[0]
    sem=pd.DataFrame([
        ['BACC','(正类召回率＋负类召回率)/2','(NPV＋PPV)/2，即macro precision','不能直接当作同一定义比较'],
        ['Recall','TP/(TP+FN)','TN/(TN+FN)，即NPV','作者打印值不是正类召回率'],
        ['Precision','TP/(TP+FP)','TN/(TN+FP)，即specificity','作者打印值不是正类精确率'],
        ['F1','2TP/(2TP+FP+FN)','2TN/(2TN+FP+FN)，即负类F1','正负类不同'],
        ['MCC','[-1,1]，综合四格统计','TP/TN名称交换不改变该公式值','公式可比，但数据版本/划分仍不同'],
        ['ROC-AUC','连续得分区分正负类的能力','使用decision_function或正类概率','公式可比，但数据版本/划分仍不同']
    ],columns=['指标','本次标准定义','作者helper实际计算','比较限制'])
    env=md(pd.read_csv(OUT/'environment_comparison.csv'))
    ba=entry('A_author_cleaned_code_grid','SVM','balanced_accuracy');mc=entry('A_author_cleaned_code_grid','SVM','mcc')
    fba=entry('F_TableS2_fixed_parameters','SVM','balanced_accuracy');fmc=entry('F_TableS2_fixed_parameters','SVM','mcc')
    text=f'''# 两次完整复现、论文与作者代码结果对照

第二次实际执行日期：2026-09-23。开始/结束时间：`{check['actual_run2_status']['start_utc']}` / `{check['actual_run2_status']['end_utc']}`（UTC）。第一次产物日期：2026-09-22，来源于已保存的原始复现实验，第二次从原始序列重新计算，未读取第一次训练结果作为答案。

## 核心结论

两次各有131条模型/划分评估记录，全部一一配对。数据和依赖版本一致；ESM逐位相同：**{check['esm_bitwise_equal']}**；CleanLab删除决策相同：**{check['cleanlab_keep_masks_equal']}**；训练/测试成员相同：**{check['split_membership_identical']}**。标准指标出现超过1e-12变化的实验/模型：**{changes}**。

SVM A方案两次标准BACC均值分别为{ba.run1_mean:.6f}、{ba.run2_mean:.6f}，MCC分别{mc.run1_mean:.6f}、{mc.run2_mean:.6f}。最终部署SVM对同一420条序列的分类差异为{check['svm_all420_disagreements']}条，最大decision score差异{check['svm_all420_max_score_difference']:.9g}。这项检查衡量两模型行为一致性，不能作为新的独立测试成绩。

科学复现结论仍为 **C. Partially reproduced**：重复运行稳定与论文严格复现成功是两回事。作者CleanLab源程序/环境未公开，存储Notebook数据版本不同，指标命名和调参评分不一致，而且原设计存在评估泄漏风险。

## 1. 先认清四种对照对象

1. **第一次、第二次重跑**：相同当前公开数据、相同环境、相同代码和参数协议，可逐seed配对比较。
2. **作者Notebook保存输出**：历史结果；其中展示过343条数据，当前清洗文件339条。不是作者训练权重，也不是我们本次运行得到的结果。
3. **论文正文/Table S2数值**：论文声称的结果。Table S2标题写10折CV，正文说十次随机划分，语义有歧义。
4. **F：按论文固定参数重跑**：采用S2给出的参数，在当前339条数据上按公开seeds0…9评估。没有为了追近论文结果再调参。

主仓库未提供原始训练权重，因此无法将作者原权重和本次模型逐权重比较。提供的原始文件是公开数据、Notebook和评估helper。

## 2. 全部模型：两次结果与原结果

均值±标准差是**每次运行内十个划分**的样本标准差（ddof=1），不是“第一次和第二次之间的标准差”。MCC变化=第二次均值−第一次均值。五个传统模型使用A方案，LSTM使用E方案的十次独立划分；作者保存的LSTM MCC只有split42单次，不能把它当作十次均值。

{md(main)}

![两次结果与论文均值对比](two_runs_vs_paper.png)

图中误差条为各次运行内十个划分的样本标准差；论文仅显示均值。

作者保存MCC与当前结果的差异不能只归因于随机性：输入版本、超参数和环境都有差异。其BACC/Recall/Precision还存在定义错误，因此没有把这些历史打印值直接混进本表的标准BACC列。完整同公式比较见 `author_same_formula_comparison.csv`；所有九项标准指标及131对逐seed差值见 `paired_seed_results.csv`。

## 3. 第二次有没有改变参数或选种子？

没有。原始网格、评分、初始化约定、划分种子和CleanLab规则保持不变。只对第二次副本的common.py改动两行路径设置，以独立保存输出；diff与逐脚本SHA在第二次logs中。作者原始文件没有修改，第一次快照的170项文件SHA全部通过验证。

{md(selected[['model','run1_parameters','run2_parameters','parameters_equal','run1_cv_accuracy','run2_cv_accuracy']])}

随机森林原网格始终保留 `random_state=None`。它的首次/第二次最佳参数分别为 `{rfc.run1_parameters}` / `{rfc.run2_parameters}`。这种设置允许并行CV及重新fit产生随机波动；设置numpy全局seed并不等价于为每个RF estimator固定随机状态。与之相对，F论文固定参数独立实验里的RF显式设random_state=0。两种协议没有混在一起，也未用更好的一次替换较差的一次。

## 4. SVM四条实验路线

{md(svm)}

论文SVM：BACC=0.963±0.022、Recall=0.976±0.024、Precision=0.951±0.030、MCC=0.927±0.044。第二次A方案标准BACC与论文均值相差{ba.run2_mean-.963:+.6f}，MCC相差{mc.run2_mean-.927:+.6f}。第二次F方案BACC={fba.run2_mean:.6f}、MCC={fmc.run2_mean:.6f}，分别与论文相差{fba.run2_mean-.963:+.6f}、{fmc.run2_mean-.927:+.6f}。

- A：公开339条清洗特征＋源码的accuracy评分网格，SVM为C=30、poly、degree=5、tol=1e-5。
- B：仅把A的网格评分改为MCC；当前选择相同参数，并非两套不同最终模型。
- C：420条序列重新编码＋独立CleanLab保留353条，之后重新网格训练；不是作者缺失清洗代码的精确重跑。
- F：论文固定C=10、poly、degree=1、tol=1e-5；不重新选择参数。
- 作者Notebook保存的SVM为C=30、rbf、degree=1，MCC打印0.550±0.072；它与上述各路线的参数/数据并不相同。

## 5. 总流程图

```mermaid
{flow}```

## 6. 按步骤解释：输入什么、做什么、得到什么

{md(pd.DataFrame(stepfacts))}

**ESM为什么有320维？** 每条肽有若干氨基酸。模型第6层给每个残基一个320维表示；沿残基方向求平均，把不同长度的肽变成同样长度的320维向量。BOS、EOS、padding不参与均值；不做额外归一化。这个过程使用预训练模型，不是用420条数据重新训练ESM。

**CleanLab到底做了什么？** 先让LR对每条肽给出“在未见过该条训练数据的CV模型下”的预测概率，再结合现有标签识别可疑样本。主重建固定5折、seed0、max_iter5000和默认过滤，不反复试阈值。两次都保留353条，和作者339条的保留集合重叠333条，420条保留/删除决策一致率93.81%。数量一致本身不足以证明算法一致；这里连数量也不等，且作者源码缺失。

**什么叫重新跑一遍？** 第二次重新执行了Notebook尝试、ESM前向推理、CV预测概率、CleanLab决策、全部网格和LSTM训练、最终模型保存及验证。仅复用已有ESM预训练权重缓存；没有复用第一次的embedding文件、清洗mask或已训练分类器。

## 7. 训练步骤与泄漏为什么必须分开看

```mermaid
{training_flow}```

原方法先用完整数据清洗，再划分；样本保留决策已使用测试标签信息。然后先选一次超参数，再把全体339条重新划分十次，使后续测试集有52–64/68条参加过此前调参。原helper还反复fit同一个estimator，导致Notebook随后打印的初始留出成绩使用了被重新训练过的模型。

重复划分无完全重复肽，但每次68条测试肽中有12–17条与训练肽的normalized Levenshtein similarity>0.8。相似度定义为1−编辑距离/max长度；审计使用严格大于号，不把等于0.9计为>0.9。

本次继续原设计以检验重复性，没有悄悄改成无泄漏方案。两次分数相同不会修复上述问题，也不能据此证明真实外部泛化或肽的生物学功效。

## 8. 为什么我报告的指标与作者打印值不能直接对号？

标准二分类矩阵为 `[[TN,FP],[FN,TP]]`；作者代码按`TP,FP,FN,TN`接收，交换了TN和TP的名称。

{md(sem)}

BACC标准含义是“对正例、负例各给一半权重的识别率”；Recall是有活性标签的肽中找回多少；Precision是判为阳性的肽中真阳性标签占多少；MCC综合考虑四种预测结果，1为完全一致，0附近表示相关性弱，负值表示反向关联。

## 9. 证据、边界和复跑入口

{env}

- 原始作者输入完整性：`input_integrity.csv`。
- 完整逐seed配对：`paired_seed_results.csv`，126组实验/模型/标准指标汇总：`model_repeatability_summary.csv`。
- 每个网格候选的两次得分：`paired_grid_scores.csv`；最佳参数：`selected_parameters_comparison.csv`。
- 420条embedding/清洗决策配对：`embedding_cleanlab_paired_samples.csv`。
- 作者原打印定义对比：`author_same_formula_comparison.csv`；论文与两次结果：`paper_vs_both_runs.csv`。
- 两个最终SVM的预测诊断：`svm_prediction_agreement_420.csv`。
- 第一次目录：`../../runs/run_01_2026-09-22/`；第二次目录：`../../runs/run_02_2026-09-23/`。
- 本报告及流程图由 `reproduction/src/compare_runs.py` 从实际结果生成，执行 `.venv/bin/python reproduction/src/compare_runs.py` 可以重新生成对比，不会重训模型。

两次运行不构成统计显著性检验；十个测试划分也互有重叠。这里报告实际差值和逐seed一致性，不计算误导性的独立样本p值。RF随机性保留是忠实于原代码的选择，并非遗漏其风险。未公开的原训练权重与CleanLab源码仍是严格科学复现的缺口。
'''
    (OUT/'TWO_RUN_COMPARISON.md').write_text(text)
    # Standalone scientific figures, derived from real saved metrics.
    names=['LR','RF','KNN','SVM','LightGBM','LSTM'];x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,metric in zip(axes,['balanced_accuracy','mcc']):
        v=[]
        for name in names:
            exp='E_independent_LSTM_10_splits' if name=='LSTM' else 'A_author_cleaned_code_grid';v.append(entry(exp,name,metric))
        ax.errorbar(x-.11,[r.run1_mean for r in v],yerr=[r.run1_std for r in v],fmt='o',capsize=3,label='Run 1 (mean +/- sample SD)',color='#177b8c')
        ax.errorbar(x+.11,[r.run2_mean for r in v],yerr=[r.run2_std for r in v],fmt='x',capsize=3,label='Run 2 (mean +/- sample SD)',color='#a54623')
        ax.scatter(x,[refs[(refs.source=='paper_Table_S2')&(refs.model==n)&(refs.metric==metric)].iloc[0]['mean'] for n in names],marker='D',s=25,color='#666666',label='Paper mean only')
        ax.set_xticks(x,names);ax.set_ylabel(metric.replace('_',' ').upper());ax.set_ylim(.68,1.03);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Run 1 vs Run 2: same inputs and protocol; evaluation leakage caveats remain',fontsize=12)
    fig.savefig(OUT/'two_runs_vs_paper.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,['balanced_accuracy','mcc']):
        for name in names:
            exp='E_independent_LSTM_10_splits' if name=='LSTM' else 'A_author_cleaned_code_grid'
            r1=a[(a.experiment==exp)&(a.model==name)].set_index('seed');r2=b[(b.experiment==exp)&(b.model==name)].set_index('seed')
            ax.plot(range(10),(r2[metric]-r1[metric]).sort_index(),marker='o',markersize=3,alpha=.8,label=name)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xlabel('Matched split seed');ax.set_ylabel('Run 2 - Run 1: '+metric);ax.set_xticks(range(10));ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Paired differences (zero means identical metric, not independent validation)',fontsize=11)
    fig.savefig(OUT/'paired_seed_differences.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
