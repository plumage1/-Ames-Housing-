# 房价预测项目 README

## 1. 项目简介

本项目基于 Ames Housing 数据集完成房价预测任务，内容覆盖：

- 原始数据整理
- 数据预处理
- 模型训练与评估
- 新样本预测
- 结果可视化

仓库中已包含：

- 原始数据文件
- 预处理后数据文件
- 完整 Python 源码
- README 运行说明
- 数据说明文档
- 模型结果与可视化图表

## 2. 运行环境

- Python `3.9+`
- 本项目已在 Python `3.13` 环境下验证通过
- 建议在 Windows PowerShell 中运行

## 3. 依赖库与安装方法

项目依赖如下：

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

安装步骤：

```powershell
python -m venv .venv
python -m pip --python .venv\Scripts\python.exe install -r requirements.txt
```

激活虚拟环境：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. 运行步骤

请在根目录下按以下顺序运行：

```powershell
.\.venv\Scripts\python.exe code\01_data_exploration.py
.\.venv\Scripts\python.exe code\02_data_visualization.py
.\.venv\Scripts\python.exe code\03_data_preprocessing.py
.\.venv\Scripts\python.exe code\04_model_training.py
.\.venv\Scripts\python.exe code\05_prediction.py
```

各脚本功能说明：

- `code/01_data_exploration.py`：读取原始训练集，输出数据规模、字段信息和基本统计量
- `code/02_data_visualization.py`：生成原始数据分布图和关系图
- `code/03_data_preprocessing.py`：完成缺失值处理、异常值裁剪、分类变量编码和标准化，并保存预处理结果
- `code/04_model_training.py`：比较线性回归与随机森林模型，使用交叉验证进行模型选择，并输出独立 holdout 测试结果
- `code/05_prediction.py`：加载最优模型，对新样本进行预测并输出预测可视化结果

## 5. 数据与结果文件说明

### 5.1 数据文件

- `data/raw/train.csv`：原始训练集
- `data/raw/test.csv`：原始测试集
- `data/processed/X_train.csv`：预处理后的训练特征
- `data/processed/y_train.csv`：训练目标值
- `data/processed/train_processed.csv`：合并后的预处理训练数据
- `data/processed/X_test_processed.csv`：预处理后的测试特征

### 5.2 文档文件

- `docs/DATA_DESCRIPTION.md`：数据来源、字段示例、预处理方法说明
- `docs/preprocessing_summary.csv`：各字段缺失值处理、异常值处理和缩放摘要

### 5.3 输出文件

- `output/best_model.pkl`：最终选出的最佳模型
- `output/LinearRegression_model.pkl`：线性回归模型
- `output/RandomForest_model.pkl`：随机森林模型
- `output/preprocessing_artifacts.pkl`：预处理规则文件
- `output/feature_columns.csv`：训练阶段特征列清单
- `output/prediction_input_raw.csv`：预测输入原始样本
- `output/prediction_input_processed.csv`：预测输入预处理结果
- `output/prediction_result.csv`：预测结果

### 5.4 结果图表与评估文件

- `result/01_data_visualization.png`：原始数据可视化图
- `result/model_metrics.csv`：模型评估汇总表
- `result/cv_fold_metrics.csv`：交叉验证各折结果
- `result/model_comparison.png`：模型对比图
- `result/actual_vs_predicted.png`：真实值与预测值对比图
- `result/residual_distribution.png`：残差分布图
- `result/test_set_predictions.csv`：测试集预测结果
- `result/prediction_visualization.png`：新样本预测可视化图

## 6. 模型方法说明

本项目当前采用以下建模流程：

1. 从原始训练集读取数据
2. 划分模型选择集与独立 holdout 测试集
3. 在训练部分内部完成预处理拟合，避免数据泄漏
4. 使用 5 折交叉验证比较不同模型
5. 根据交叉验证结果选择最佳模型
6. 在完整训练数据上重新训练最佳模型并保存

参与比较的模型：

- `LinearRegression`
- `RandomForestRegressor`

当前结果中，最佳模型为 `RandomForest`。

## 7. 当前验证结果

在当前代码和数据下，项目已成功运行完成。主要结果如下：

- 最佳模型：`RandomForest`
- 交叉验证平均 `R2`：约 `0.8454`
- 独立 holdout 测试集 `R2`：约 `0.8987`

以上结果可在 `result/model_metrics.csv` 中查看。

## 8. 目录结构

```text
project2/
|-- Readme.md
|-- requirements.txt
|-- code/
|   |-- 01_data_exploration.py
|   |-- 02_data_visualization.py
|   |-- 03_data_preprocessing.py
|   |-- 04_model_training.py
|   |-- 05_prediction.py
|   |-- preprocessing_utils.py
|   |-- project_paths.py
|-- data/
|   |-- raw/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- processed/
|       |-- X_train.csv
|       |-- y_train.csv
|       |-- train_processed.csv
|       |-- X_test_processed.csv
|-- docs/
|   |-- DATA_DESCRIPTION.md
|   |-- preprocessing_summary.csv
|-- output/
|   |-- best_model.pkl
|   |-- best_model_name.txt
|   |-- feature_columns.csv
|   |-- LinearRegression_model.pkl
|   |-- RandomForest_model.pkl
|   |-- preprocessing_artifacts.pkl
|   |-- prediction_input_raw.csv
|   |-- prediction_input_processed.csv
|   |-- prediction_result.csv
|-- result/
|   |-- 01_data_visualization.png
|   |-- model_metrics.csv
|   |-- cv_fold_metrics.csv
|   |-- model_comparison.png
|   |-- actual_vs_predicted.png
|   |-- residual_distribution.png
|   |-- test_set_predictions.csv
|   |-- prediction_visualization.png
```

## 9. 提交说明

本项目已整理为可直接提交版本。若需检查结果，可直接查看：

- `docs/DATA_DESCRIPTION.md`
- `result/model_metrics.csv`
- `result/` 下的图表文件

若需要复现，可按第 4 部分步骤重新运行全部脚本。
