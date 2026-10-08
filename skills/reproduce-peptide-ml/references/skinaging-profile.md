# 抗皮肤衰老肽论文专用配置

仅用于 Zhang et al. (2025), *Discovering potential anti-skin-aging peptides in collagen: computer-assisted rapid screening and structure–activity relationships*，DOI [10.1186/s42825-025-00215-8](https://doi.org/10.1186/s42825-025-00215-8)。

作者仓库：[RH0627/skinaging_predictor_ZJU](https://github.com/RH0627/skinaging_predictor_ZJU)，已审计版本 `7d7b6b55256018d1718d8a4dd4cc39faa86e45f1`。以下是2026-09-22/23两次执行的事实，后续运行必须重新检查，不能直接复制为新成绩。

## 项目入口

相对于当前工作区：

- `skinaging_predictor_ZJU/`：作者原始仓库，保持不变。
- `reproduction/run_reproduction.py`：13阶段完整执行入口。
- `reproduction/src/`：数据、ESM、清洗、模型、LSTM、泄漏、报告与验证脚本。
- `reproduction/requirements.txt`：该次兼容环境的锁文件，不是作者原环境。
- `reproduction/sources/`：论文HTML、补充DOCX及提取表格。
- `reproduction/runs/run_01_2026-09-22/` 和 `run_02_2026-09-23/`：已完成的两次实验。
- `reproduction/comparisons/run_01_vs_run_02/TWO_RUN_COMPARISON.md`：此前完整对照。

若工作区没有这些执行脚本，先定位用户已有项目或根据作者代码实现执行层；不要假定安装本 skill 就包含训练数据、模型权重或完整训练环境。

旧 `repeat_experiment.py` 与 `compare_runs.py` 固定引用前两次目录。不要用它们覆盖旧实验，也不要改日期后把旧结果解释为新运行。任意新 run 用本 skill 的目录准备工具；数值配对用通用比较工具，完整报告根据新数据生成。

## 数据、表示与清洗

| 项目 | 已验证配置 |
|---|---|
| 原始数据 | `Database/Oringinal_data.xlsx`，420条，210阳性/210随机对照 |
| 作者清洗数据 | `X_320_clean_LR.csv` 与 `y_after_clean.csv`，339条，161阳性/178对照 |
| 身份映射 | 特征CSV索引 `No` 对应原始 `No`，并核对标签 |
| ESM | `esm2_t6_8M_UR50D`，layer6，320维，残基均值，排除BOS/EOS/padding |
| 推理 | CPU、batch32、eval/no_grad、float32，无额外归一化 |
| 独立CleanLab | LR(max_iter=5000)，StratifiedKFold(5, shuffle=True, random_state=0)，OOF predict_proba |
| 清洗过滤 | CleanLab 2.7.0，prune_by_noise_rate 默认规则；self_confidence为质量分数 |

作者清洗实现未公开。上述重建保留353条，其中170阳性/183对照；与作者保留集合交集333，作者独有6，重建独有20。不要把353强行调成339。

此前重算特征与作者向量平均余弦相似度约0.9999999999993、最大绝对差约3.8e-6；两次自身重算逐位相同。这些数值用于历史对照，不是所有设备的硬性验收阈值。

## 实验协议

| 标识 | 数据与操作 |
|---|---|
| A_author_cleaned_code_grid | 作者339条；从原Notebook提取完整网格，默认accuracy评分 |
| B_author_cleaned_MCC_grid | 相同339条和SVM网格，仅改MCC评分 |
| C_independent_ESM_CleanLab_code_grid | 原始420条重算ESM，独立清洗保留353条，再做SVM网格 |
| D_author_LSTM_split42_harness_seed0 | 作者LSTM split42，补充初始化seed0 |
| E_independent_LSTM_10_splits | LSTM splits0–9，各次初始化对应seed |
| F_TableS2_fixed_parameters | 作者339条，固定论文S2参数，不重新搜索 |

传统模型首次调参划分seed2001、test_size=.2、无stratify；训练271条做10折CV。选参后重新在全体数据上按seeds0–9做80/20划分。353条路线按自身样本数计算人数，不套用271/68。

五种传统模型：LR、RF、KNN、SVM、LightGBM；网格从作者原Notebook完整提取，不以历史最佳值替代搜索。原代码RF保持 `random_state=None`，全局NumPy seed0不能保证并行RF重复一致。

论文固定参数：

- LR：C=1，class_weight=balanced，max_iter=5000，solver=saga；执行层补充random_state=0。
- RF：max_depth=4，max_features=sqrt，n_estimators=160；执行层补充random_state=0。
- KNN：algorithm=auto，leaf_size=10，n_neighbors=3，weights=uniform。
- SVM：C=10，kernel=poly，degree=1，tol=1e-5。
- LightGBM：boosting=gbdt，learning_rate=.5，n_estimators=40，objective=binary，reg_lambda=0。

LSTM：128/64/32单元层，Dense64/10/1，Dropout .15，Adam，100epochs，batch32。该输入把320维向量重塑为单时间步，不是逐残基序列输入。Keras3曾要求标签从(n,)改为(n,1,1)；保存错误和最小形状修复，不调整网络追分。

## 已知解释陷阱

- 原Notebook原样失败；仅修路径后ESM成功，训练Notebook在未定义的X_out处中断。完整网格通过执行层运行。
- 作者helper误把 `confusion_matrix.ravel()` 当作 `TP,FP,FN,TN`；实际打印Recall=NPV、Precision=specificity、F1=负类F1、BACC=(NPV+PPV)/2。MCC和ROC-AUC公式未因此改变。
- 当前清洗数据339条；历史Notebook展示过343条。原训练权重未公开，不能声称直接预测过作者原模型。
- 论文称按MCC调参，代码默认accuracy；S2标题称十折CV，正文称十次随机划分，协议存在歧义。
- 先清洗后划分；后续每个68条测试集中52–64条参与过此前调参。另有12–17条与训练肽的normalized Levenshtein similarity严格大于.8。
- 相似度为 `1 - edit_distance / max(length1, length2)`。不要把>.8、>=.8或其他算法混用。

## 历史对照及环境

两次各131条评估记录；仅A/RF的标准指标变化。A/SVM两次BACC=.957879、MCC=.915383；F/SVM两次BACC=.930126、MCC=.859924；论文为BACC=.963、MCC=.927。原Notebook保存SVM是RBF、C30，MCC=.550，不能与当前poly网格结果当作同条件提升。

Python3.12.14，torch2.6.0、fair-esm2.0.0、scikit-learn1.5.2、cleanlab2.7.0、numpy1.26.4、pandas2.2.3、lightgbm4.6.0、tensorflow2.18.0、keras3.15.1。作者记录Python3.9；现代环境兼容重跑与原环境重现要区分。

macOS的LightGBM曾复用环境内PyTorch的libomp，避免改系统库；不要在其他平台机械套用此补丁。先检查解释器真实可用性，不假定macOS的系统Python占位入口能执行。

已得结论为 **C. Partially reproduced**。新运行应根据新证据判断，而不是照抄该结论。
