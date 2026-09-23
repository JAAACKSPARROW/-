# 科研复现报告 / Reproduction report

## 1. Paper

Ruihao Zhang, Yang Li, Yonghui Li, Hui Zhang (2025). *Discovering potential anti-skin-aging peptides in collagen: computer-assisted rapid screening and structure–activity relationships*. Collagen and Leather 7:30. DOI: [10.1186/s42825-025-00215-8](https://doi.org/10.1186/s42825-025-00215-8). 本任务复现计算建模部分；没有执行分子合成、湿实验或重新完成整篇论文的分子对接。

## 2. Repository

[作者仓库](https://github.com/RH0627/skinaging_predictor_ZJU)，branch `main`，commit `7d7b6b55256018d1718d8a4dd4cc39faa86e45f1`。作者提交日期 2025-01-27，早于论文发表。原始仓库独立保留，根项目用 gitlink/submodule 固定其版本。`data_audit/checksums.txt` 记录所有原文件 SHA256。

WORKSPACE_ROOT=`/Users/jingyuan/Documents/ChatGPT/模型构建`。所有代码、结果、缓存和环境均在此目录；临时的早期库启动缓存不属于结果，后续已设置工作区缓存路径。

## 3. Environment

```json
{
  "torch": "2.6.0",
  "fair-esm": "2.0.0",
  "scikit-learn": "1.5.2",
  "cleanlab": "2.7.0",
  "numpy": "1.26.4",
  "pandas": "2.2.3",
  "lightgbm": "4.6.0",
  "tensorflow": "2.18.0",
  "keras": "3.15.1",
  "scipy": "1.14.1",
  "nbclient": "0.11.0",
  "python": "3.12.14",
  "os": "macOS-26.6-arm64-arm-64bit",
  "architecture": "arm64"
}
```

Apple M5 / Metal 可用，本实验使用 CPU，不使用 CUDA/MPS。系统 `/usr/bin/python3`、`/usr/bin/git` 是缺少 Xcode CLT 的不可用入口，实际使用已验证的独立运行时创建 `.venv`。作者 Python 3.9；此次 Python 3.12.14。完整锁文件 `requirements.txt`；原仓库没有依赖锁文件，不能声称环境精确一致。

## 4. Repository structure

`repository_tree.txt`、`repository_inventory.csv` 枚举全部非 .git 文件（包括作者已跟踪的 notebook checkpoint）。主要输入只有原始 XLSX、清洗特征 CSV、清洗标签 CSV；两个正式 Notebook、一个评估 helper 和十条胶原蛋白文本序列。没有公开 CleanLab 源程序、原始 420 条 embedding 文件或训练好的模型。

## 5. Data audit

原始 420 条，210 正例 / 210 随机对照；清洗后 339 条，161 / 178。原始与清洗数据均无缺失、重复序列、标签冲突、非法字符或异常标签。序列长度 2–80；完整分布见审计 CSV。随机序列的 label=0 是作者设置的对照标签，不能据此认定所有随机序列已经实验证明没有活性。

`No` 与原始 XLSX 的 `No` 精确连接，清洗后标签逐行相符。所有比较以此身份映射，不依赖向量相似度猜测配对。

## 6. Paper vs public data

当前公开文件样本数量与论文一致。但训练 Notebook 的保存输出显示 `(343,320)`、标签 `(343,)`，前五条 embedding 也不同；ESM Notebook 保存的输入序列数为 1564，且输入文件是占位 Windows 路径。这些保存输出不能当作当前 420/339 数据的运行证据。

## 7. ESM-2 implementation

已从实际代码确认 checkpoint、layer、pooling、CPU、eval/no_grad、无归一化。详见 `ESM2_IMPLEMENTATION.md`。路径修复后的完整原 ESM Notebook 已在 CPU 执行，原始文件未更改。

## 8. Embedding reproduction

重新生成全部 `(420,320)` float32 embedding。与 339 条作者特征比较：平均 cosine=0.999999999999268，中位数=0.999999999999398，最小值=0.999999999994585，mean MAE=3.28310779e-07，max absolute difference=3.78470425e-06。此误差规模支持实现一致，不能声称逐 bit 一致。逐样本数值保存于 `embedding_comparison.csv`。

## 9. CleanLab reproduction

**作者清洗代码和版本未公开，无法严格复现。** 独立重建固定使用 CleanLab 2.7.0、LR(max_iter=5000)、5 折 StratifiedKFold(shuffle=True, random_state=0)、out-of-fold predict_proba、prune_by_noise_rate 默认过滤，不修改标签或阈值。清洗作用于全部原始数据，再划分，延续论文设计，且保留相应泄漏风险。

保留 353 条（正 170 / 负 183）；与作者共同保留 333，author only 6，reproduction only 20，所有420条的决策一致率 93.809524%。`issue_score` 是 given-label self-confidence，越低越可疑，不是概率校准结果。明细包含 p(class=1)、given-label probability 和保留/删除标记。

独立敏感性分析事先固定 seeds=0…9，另检查10折与LR默认100次迭代，结果保存在 `cleanlab_sensitivity.csv`。默认实验始终是 seed0 的353条，未挑选接近339的结果。由于缺少原版本/CV/seed/API，不能把差异单独归因于某一个版本。

## 10. ML models

原始 Notebook 无修改执行均 FAIL（Windows路径/不存在的相对路径）；路径修复后 ESM PASS，训练 PARTIAL，在 LR 单元遇到未定义 `X_out`。完整 traceback、修复 diff 和执行副本均保留。随后从原 Notebook 提取并执行五个模型的真实完整网格，保留原网格顺序、10折、默认 scoring，不削减候选。

LSTM 使用作者网络、100 epochs、batch32，现代 Keras 要求目标维度与 `(n,1,1)` 输出一致，最小修复只 reshape 标签。原始错误也保留。原代码只有 split42，独立重复实验使用0…9划分/初始化种子，必须与原始方法区分。

以下均使用标准指标定义、正类=1、样本标准差 ddof=1。PR-AUC 列采用 average precision，未冒称梯形积分。详细 fit/predict 时间、参数和划分大小见 `model_metrics.csv`。

| experiment | model | balanced_accuracy | precision | recall | mcc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| A_author_cleaned_code_grid | KNN | 0.9038 ± 0.0408 | 0.8997 ± 0.0452 | 0.9068 ± 0.0502 | 0.8084 ± 0.0812 | 0.9410 ± 0.0261 |
| A_author_cleaned_code_grid | LR | 0.9458 ± 0.0316 | 0.9446 ± 0.0391 | 0.9471 ± 0.0379 | 0.8916 ± 0.0638 | 0.9904 ± 0.0086 |
| A_author_cleaned_code_grid | LightGBM | 0.9400 ± 0.0166 | 0.9426 ± 0.0170 | 0.9350 ± 0.0350 | 0.8797 ± 0.0341 | 0.9876 ± 0.0083 |
| A_author_cleaned_code_grid | RF | 0.9388 ± 0.0300 | 0.9649 ± 0.0185 | 0.9098 ± 0.0491 | 0.8784 ± 0.0598 | 0.9850 ± 0.0098 |
| A_author_cleaned_code_grid | SVM | 0.9579 ± 0.0323 | 0.9616 ± 0.0436 | 0.9529 ± 0.0350 | 0.9154 ± 0.0643 | 0.9918 ± 0.0081 |
| B_author_cleaned_MCC_grid | SVM | 0.9579 ± 0.0323 | 0.9616 ± 0.0436 | 0.9529 ± 0.0350 | 0.9154 ± 0.0643 | 0.9918 ± 0.0081 |
| C_independent_ESM_CleanLab_code_grid | SVM | 0.9530 ± 0.0192 | 0.9322 ± 0.0420 | 0.9702 ± 0.0236 | 0.9053 ± 0.0395 | 0.9942 ± 0.0049 |
| D_author_LSTM_split42_harness_seed0 | LSTM | 0.9268 (single split) | 0.9412 (single split) | 0.9143 (single split) | 0.8533 (single split) | 0.9584 (single split) |
| E_independent_LSTM_10_splits | LSTM | 0.9450 ± 0.0306 | 0.9480 ± 0.0432 | 0.9432 ± 0.0326 | 0.8914 ± 0.0624 | 0.9840 ± 0.0169 |
| F_TableS2_fixed_parameters | KNN | 0.9112 ± 0.0292 | 0.9112 ± 0.0527 | 0.9132 ± 0.0439 | 0.8247 ± 0.0590 | 0.9478 ± 0.0257 |
| F_TableS2_fixed_parameters | LR | 0.9297 ± 0.0334 | 0.9294 ± 0.0354 | 0.9296 ± 0.0474 | 0.8594 ± 0.0665 | 0.9878 ± 0.0102 |
| F_TableS2_fixed_parameters | LightGBM | 0.9419 ± 0.0150 | 0.9461 ± 0.0268 | 0.9359 ± 0.0335 | 0.8853 ± 0.0291 | 0.9867 ± 0.0091 |
| F_TableS2_fixed_parameters | RF | 0.9357 ± 0.0326 | 0.9512 ± 0.0347 | 0.9183 ± 0.0391 | 0.8713 ± 0.0662 | 0.9848 ± 0.0104 |
| F_TableS2_fixed_parameters | SVM | 0.9301 ± 0.0314 | 0.9258 ± 0.0389 | 0.9320 ± 0.0511 | 0.8599 ± 0.0614 | 0.9863 ± 0.0096 |

## 11. SVM reproduction

| source | details |
| --- | --- |
| Paper | Table S2: C=10, degree=1, kernel=poly, tol=1e-5; MCC scoring claimed; gamma unspecified |
| GitHub source | C=[.001,.01,.1,.5,1.1,1.3,5,10,30,50,100,300]; kernel=[linear,poly,rbf,sigmoid]; degree=[1,3,5,7,9]; tol=1e-5; gamma=scale default; scoring omitted |
| GitHub saved output | C=30,degree=1,kernel=rbf,tol=1e-5; BACC .773±.037; MCC .550±.072; underlying data shown as 343 rows |
| Current A | {'C': 30, 'degree': 5, 'kernel': 'poly', 'tol': 1e-05} |
| Current B (MCC) | {'C': 30, 'degree': 5, 'kernel': 'poly', 'tol': 1e-05} |

共240候选、每候选10折。degree 对非 poly 核不起作用，仍保留作者重复候选。A 使用当前作者清洗特征 + 原代码 accuracy 评分；B 只将网格 scoring 改为 MCC，其他保持相同。本次 A/B 选中相同参数，因此重复评估结果相同。C 为重新 embedding + 独立 CleanLab + 原网格，不能称为作者原始清洗的精确重跑。另做 F：固定 Table S2 参数 C=10、degree=1、poly、tol=1e-5，直接重复0…9划分，不重新调参。Table S2 五个ML模型均另以论文固定参数评估，RF/LR额外固定random_state=0并明确属于独立实验。

最终 `models/svm_final.joblib` 以 A 的参数在全部339条清洗数据 refit，供预测使用。报告测试指标来自各次271/68划分，绝不使用最终全数据模型评价这些测试集。没有 scaler。预测输出 decision_function score，非概率。

## 12. Paper vs reproduced metrics

论文 benchmark 取自正文3.1.2；没有以该数值为调参目标。标准 Recall/Precision 与原 helper 的同名打印值语义不一致，因此差异表是数值比较，不能作为严格同定义的优劣检验。

| metric | paper_mean | paper_std | reproduced_mean | reproduced_std | absolute_difference |
| --- | --- | --- | --- | --- | --- |
| balanced_accuracy | 0.963000 | 0.022000 | 0.957879 | 0.032325 | 0.005121 |
| precision | 0.951000 | 0.030000 | 0.961555 | 0.043597 | 0.010555 |
| recall | 0.976000 | 0.024000 | 0.952942 | 0.035016 | 0.023058 |
| mcc | 0.927000 | 0.044000 | 0.915383 | 0.064257 | 0.011617 |

Table S2 固定 SVM 参数 F 的标准 BACC 为0.930126±0.031354、MCC为0.859924±0.061380，与正文值的绝对差分别0.032874、0.067076；此实验没有再调参。其他 B/C/F 实验的完整均值、标准差、绝对/相对差异见 `paper_vs_reproduction.csv`。这十次测试集互有重叠，标准差不是独立样本估计的置信区间；不据此作显著性声明。

## 13. Reproducibility across seeds

作者 `self_function.py` 明确公开 split seeds=0…9，`random_state=i`、test_size=.2、无 stratify；并非种子未公开。网格首次训练划分 seed2001，cv=10 默认 StratifiedKFold、不 shuffle。正文明确说十次随机划分，但补充 Table S2 标题写“10-fold cross-validation”，而给出的SVM数值与正文重复划分数值完全相同，存在正文/表题歧义。源码 helper 明确是十次随机划分，不能把表题视为证明这些数字来自10折。

RF 的 estimator random_state 和 LSTM 初始权重种子未公开。原 RF 保留 random_state=None；harness 设置 numpy seed0，但并行 worker 的未固定 RNG 仍限制逐位可重复性。独立 LSTM 明确设置每次 seed。所有 CV 表、分割成员和种子结果保留。

## 14. Potential Data Leakage

1. 论文明确先对全部数据 CleanLab 筛选再 split；即使清洗概率是 OOF，测试标签参与样本筛选，评估不是完全外部验证。原清洗代码缺失，无法验证作者是否真正用了 OOF；本重建明确使用 OOF。
2. 先在 seed2001 训练集选超参数，再用同一最优参数进行0…9全体数据重新划分。各次测试集有 52–64 / 68 条已参加此前超参数选择，存在调参信息重叠。详见 `hyperparameter_selection_overlap.csv`。
3. 原 helper 直接反复 fit 同一个 best_estimator 对象，返回 seed9 模型。随后 Notebook 的 grid_search.score 会用这个已被修改的 estimator 评价 seed2001 测试集，不能当作原始调参后的独立 holdout 成绩。此次独立标准指标使用 clone，另保留原 helper 输出。
4. 没有 exact sequence duplicates。采用 normalized Levenshtein similarity=1−distance/max(lengths)，严格 `>0.8` / `>0.9`。作者清洗数据十次划分每次有 12–17 / 68 条测试肽与训练肽 similarity>0.8。存在相似序列跨分割；这提示高分可能部分受相似性影响，不能仅凭此量化因果贡献。
5. 作者代码没有 scaler，因此没有证据支持 scaler 全数据 fit 泄漏。论文根据模型测试表现选定 SVM，测试集参与模型家族选择的风险需要保留。

审计只记录这些问题，不修改作者实验。未执行训练内清洗+嵌套CV+同源分组切分的独立外部验证，不能把当前高分解读为无泄漏泛化表现。

## 15. Paper-code inconsistencies

- 论文写 GridSearchCV 使用 MCC，源码未传 scoring，实际是 accuracy。
- Table S2 SVM 为 C=10/degree1/poly，与保存输出 C=30/degree1/rbf、当前重跑 C=30/degree5/poly 三者不同。LR 表列 balanced/saga，而源码默认 class_weight=None/solver=lbfgs；Table S2 的 KNN BACC 标准差写0.306，与其余指标尺度差异明显，只标注待作者澄清，不擅自改成0.0306。
- 存储输出343条，公开清洗文件339条；存储SVM核/性能也与当前数据运行不同。
- `confusion_matrix` 返回 `[TN,FP,FN,TP]`，作者按 `[TP,FP,FN,TN]` 解包。打印 Recall 实为 NPV，打印 Precision 实为 specificity，打印 F1 为负类F1，打印 BACC 实为 `(NPV+PPV)/2`（macro precision），不是 balanced accuracy。MCC 的该项重命名不改变数值；AUC 用连续分数，定义正确。
- 论文描述六种模型十次评估，公开 LSTM 单元只展示 split42 的一次评估；补充10次实验明确标记 independent。
- 声称提供全部代码，但主仓库缺少 CleanLab 实现、依赖锁和完整输入路径。

## 16. Bugs / compatibility problems

保留日志：系统 Xcode stub 不可用；沙箱初次联网失败；Jupyter 内核本地通信被拒后经授权重跑；作者 Windows 路径/相对路径错误、缺少 X_out/X_out_Sequence；现代 Keras 标签rank错误；macOS LightGBM 缺少 libomp；并行 worker 警告。早期从 stdin 启动 CleanLab multiprocessing 触发 `<stdin>` FileNotFoundError，改为有 main guard 的脚本；第一次 C 实验在清洗文件未就绪时失败，依赖就绪后重跑。失败与成功日志分别保留，不隐瞒失败实验。

## 17. Modifications made

所有新增代码位于 reproduction，作者文件 checksum 未改。修复路径/导入位置；省略未提供的外部 X_out 筛选段，仅提取现有训练网格；harness 将并行资源限制为2个worker、BLAS1线程；ESM主重算 batch32且另跑作者全batch；Keras标签reshape；LightGBM复用 `.venv` 中 PyTorch 的 libomp 而不修改系统库；标准指标另算，未悄悄修正原 helper。库/硬件/随机RNG差异可能影响模型，均未声称完全消除。

## 18. Final conclusion

**C. Partially reproduced（部分复现）。** 数据数量与标签映射核实成功，ESM实现达到数值一致，五个ML完整网格与LSTM已运行，SVM性能接近论文且可本地预测。但缺少原始CleanLab实现、公开Notebook输出与当前数据不一致、评分/指标定义差异和多种泄漏风险，无法宣布严格端到端 Fully reproduced。该评级针对科学复现一致性，不代表工程文件未交付。

Git 提交和推送状态见 `logs/git_final_state.txt`。根工作区未配置 origin 时必须报告 `NO_GITHUB_REMOTE_CONFIGURED`；作者仓库的 origin 不属于用户目标远程仓库，绝不向作者仓库推送。
