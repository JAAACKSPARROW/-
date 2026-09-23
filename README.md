# Anti-skin-aging peptide model: audited computational reproduction

本地复现 Zhang et al. (2025), **Discovering potential anti-skin-aging peptides in collagen: computer-assisted rapid screening and structure–activity relationships**, DOI [10.1186/s42825-025-00215-8](https://doi.org/10.1186/s42825-025-00215-8)。计算建模复现，不包含湿实验。

主要结论：**C. Partially reproduced**。420/339 数据数量吻合；ESM-2 向量数值一致；五种机器学习完整网格与 LSTM 已运行。作者 CleanLab 代码未公开，独立重建保留353条。存在评分定义、超参数、保存输出与数据版本不一致，以及先清洗后划分/调参重叠等泄漏风险。

原代码网格 SVM 的标准 BACC 为 **0.9579 ± 0.0323**，MCC 为 **0.9154 ± 0.0643**。论文 Table S2 固定参数独立重跑 BACC **0.9301 ± 0.0314**，MCC **0.8599 ± 0.0614**。二者不可混为同一实验。完整解释见 [报告](reproduction/REPRODUCTION_REPORT.md)。

## 第二次完整重跑与逐步流程图（2026-09-23）

第二次从原始序列重新执行全部13个阶段，最终验证通过；与第一次的131条模型/划分记录逐条配对。ESM向量、独立CleanLab决策、划分成员和SVM预测完全一致。仅原代码网格的随机森林出现波动：MCC从0.8784变为0.8826，原代码的 `random_state=None` 保持不变。科学结论仍是部分复现。

- [两次结果与论文、作者历史输出的完整对照](reproduction/comparisons/run_01_vs_run_02/TWO_RUN_COMPARISON.md)：包含流程图、12步解释、指标定义及差异原因。
- [逐模型汇总](reproduction/comparisons/run_01_vs_run_02/all_models_readable_comparison.csv)与[131条逐划分配对](reproduction/comparisons/run_01_vs_run_02/paired_seed_results.csv)。
- [第一次完整快照](reproduction/runs/run_01_2026-09-22/)与[第二次完整输出](reproduction/runs/run_02_2026-09-23/)，分别保留模型、数据、代码及日志。
- [比较验证结果](reproduction/comparisons/run_01_vs_run_02/comparison_verification.json)：第一次快照170项文件及作者输入哈希均通过检查。

重新生成对照报告与图表（不会重新训练）：

```bash
.venv/bin/python reproduction/src/compare_runs.py
```

`repeat_experiment.py` 记录本次两次实验的隔离执行方式；已有实验目录受保护。新实验应另设输出目录，以免覆盖已记录证据。

## 环境与运行

本次实测 macOS arm64、Python 3.12.14，作者记录 Python 3.9。当前锁文件为现代兼容环境，不是作者原环境。建议 Python 3.12 创建隔离环境；不要用缺少 Xcode CLT 时的 macOS `/usr/bin/python3` 占位入口。

```bash
git clone --recurse-submodules YOUR_REPRODUCTION_REPOSITORY
cd YOUR_REPRODUCTION_REPOSITORY
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/requirements.txt
.venv/bin/python reproduction/run_reproduction.py
```

当前工作区已保留完整作者 Git 仓库，无需重新下载。新克隆后若没有作者目录，执行 `git submodule update --init`。作者版本固定在 `7d7b6b55256018d1718d8a4dd4cc39faa86e45f1`。

第一次 ESM 运行需联网下载官方预训练权重，缓存只写入 `reproduction/cache/torch`。完整流程包含全部原始网格和11次LSTM训练，运行时间随设备变化。原始Notebook失败属于预期审计证据，不等于后续训练失败。

```bash
# 重用已重新生成且验证序列顺序的 ESM embedding
.venv/bin/python reproduction/run_reproduction.py --skip-esm
# 另跳过已经保留执行证据的 Notebook 阶段
.venv/bin/python reproduction/run_reproduction.py --skip-esm --skip-notebooks
# 只验证现有产物，不重跑网格
.venv/bin/python reproduction/src/verify.py
```

macOS LightGBM 的 libomp 由脚本从环境内已安装的 PyTorch 加载，不安装或修改系统库。依赖锁文件为当前平台；跨平台时应记录安装与数值差异。

## 预测

```bash
.venv/bin/python reproduction/predict.py --sequence GAPGGAGGVGEPGR
```

输出预测类和 **decision_function score（不是 probability）**。输入必须是大写标准20种氨基酸，无空格。正类表示模型预测的潜在活性，负类源于作者随机对照定义。模型是使用原代码网格选出的参数、在339条清洗数据上重新拟合的部署副本，不用于计算报告中的测试指标。

## 目录

- `skinaging_predictor_ZJU/`：完整、未改动的作者仓库（根项目以 submodule/gitlink 固定版本）。
- `reproduction/data_audit/`：校验和、样本/标签/长度审计、身份映射和分割相似性。
- `reproduction/results/`：所有实验指标、embedding、清洗决策、网格结果、科学图表。
- `reproduction/models/`：本次生成的模型，最大单个模型约数MB，纳入版本管理。
- `reproduction/executed_notebooks/`：原样失败副本、路径修复副本与已执行网格摘录。
- `reproduction/src/`：审计、执行、修复、独立分析和验证代码。
- `reproduction/logs/`：环境、原始错误、警告、执行与验证日志。
- `reproduction/sources/`：官方论文HTML、补充DOCX、来源哈希与Table S2提取。
- `.venv/`、`reproduction/cache/`：本地环境与预训练缓存，不提交。

`model_metrics.csv` 的标准指标以正类1计算；`author_printed_*` 列保留原评估函数的实际语义，不能将其当作同名标准指标。10次划分的标准差采用 ddof=1。RF原代码没有固定random_state，其并行结果不能保证逐位重现；论文固定参数独立实验另外设置随机种子。

## Git 交付

分支 `codex-reproduction`；首次复现时未配置 origin。当前发布目标为 [JAAACKSPARROW/-](https://github.com/JAAACKSPARROW/-/tree/codex-reproduction)，分支 `codex-reproduction`。本机 Git 未登录，使用已授权 GitHub 连接器上传；原始本地提交以 bundle 仅保存在本机，不公开上传。作者子仓库的 origin 不作为推送目标。提交不包含虚拟环境、ESM预训练权重或凭证，不使用force push。

`reproduction/logs/git_final_state.txt` 是本地最终提交收据，包含最终commit hash，故不纳入自身所指的提交（避免自引用哈希）；其余合理审计结果、源码、CSV、模型均进入Git。后续在本机完成 GitHub Git 身份认证后，可执行 `git push`。原始本地 commit 与通过连接器创建的发布 commit 不同；原始提交存档仅在本机 `reproduction/provenance/original_reproduction.bundle`，依用户要求不包含在公开分支中。
