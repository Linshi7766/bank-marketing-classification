# Bank Marketing · 银行营销响应预测

机器学习二分类课程实验：基于 UCI Bank Marketing 数据集，根据客户资料与电话营销联系信息，预测客户是否会购买定期存款。

## 实验流程

1. 数据读取与质量检查（含 "unknown" 缺失标记统计）
2. 探索性分析（EDA）：目标分布、年龄分布、职业购买率、通话时长分布
3. 特征预处理：类别变量 One-Hot 编码、数值变量标准化（ColumnTransformer + Pipeline）
4. 模型训练与对比：逻辑回归 vs 随机森林
5. 评估：准确率、精确率/召回率/F1、混淆矩阵、特征重要性

## 结果

| 模型 | Accuracy | Precision(yes) | Recall(yes) | F1(yes) |
|------|:---:|:---:|:---:|:---:|
| 逻辑回归（不含 duration） | 75.5% | 0.266 | 0.624 | 0.373 |
| 随机森林（不含 duration） | 86.0% | 0.419 | 0.514 | 0.462 |
| 随机森林（含 duration） | 88.2% | 0.498 | 0.759 | 0.601 |

> 类别不平衡：数据集中购买（yes）仅约 11%，故同时报告少数类的精确率/召回率/F1。

## 关键设计：识别并排除数据泄漏特征

`duration`（本次通话时长）对目标有很强的预测力 —— **含 duration 的随机森林 F1 达 0.601，
比不含时高出 0.139**。但该字段**只有通话结束后才能获知**：

> 若用于「通话前预测客户是否会购买」这一真实场景，duration 属于**事后信息**，
> 会把线下不可得的信息泄漏进模型，使评估指标虚高而模型实际不可用。

因此本实验**将 duration 从主模型移除**，仅保留含 duration 的版本作为对照，
用来量化该泄漏带来的指标虚高幅度。

## 复现

```bash
pip install pandas matplotlib seaborn scikit-learn
python CodeFile/bank_marketing_experiment.py
```

运行结果（图表与指标表）输出到 `OtherFiles/bank_marketing_outputs/`。

## 文件结构

```
CodeFile/    主程序与单元测试
Dataset/     原始数据（UCI Bank Marketing）
OtherFiles/  结果图、指标、进度管理与答辩材料
实验报告/    课程实验报告
docs/        项目文档
```

## 数据来源

Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. UCI Machine Learning Repository.
