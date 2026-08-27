# Bank Marketing Detailed Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为银行营销分类实验代码补充适合初学者复习和答辩的详细中文注释，同时保持程序行为与实验结果不变。

**Architecture:** 只修改现有单文件脚本的注释和文档字符串，不重构函数、不调整可执行语句。先记录基线并做语法检查，再分模块添加注释，最后重新执行语法检查和完整实验，将终端指标与基线进行比较。

**Tech Stack:** Python 3.13、pandas、matplotlib、seaborn、scikit-learn。

## Global Constraints

- 直接修改 `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`。
- 只增加或完善注释、文档字符串和空行。
- 不修改数据路径、模型参数、随机种子、图片名称或终端输出。
- 注释采用分块说明和关键语句解释，不对简单赋值进行机械复述。

---

### Task 1: 记录程序基线

**Files:**
- Inspect: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Create: `D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_before_comments.txt`

**Interfaces:**
- Consumes: 当前可运行的 `main() -> None` 命令行入口。
- Produces: 修改前的语法检查结果和终端输出基线。

- [ ] **Step 1: 运行语法检查**

```powershell
python -m py_compile D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py
```

Expected: 退出码为 0，无语法错误。

- [ ] **Step 2: 运行完整实验并保存基线输出**

```powershell
python D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py | Out-File -Encoding utf8 D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_before_comments.txt
```

Expected: 输出包含 `清洗前数据量: 45211 条`、三个模型场景以及前五条预测结果。

### Task 2: 增加详细中文注释

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py:1-441`

**Interfaces:**
- Consumes: 现有函数签名、参数和执行顺序。
- Produces: 函数签名与可执行语句保持不变的详细注释版脚本。

- [ ] **Step 1: 注释环境、数据与可视化模块**

在导入区、绘图配置、`load_dataset()`、`profile_and_clean()`、`_save_current_figure()` 和 `create_visualizations()` 中加入模块目的、输入输出、数据检查方法及四张图含义。

- [ ] **Step 2: 注释特征处理与模型模块**

在 `prepare_xy()`、`build_model_pipeline()` 和 `evaluate_model()` 中解释标签映射、`duration` 信息泄漏、独热编码、标准化、Pipeline、类别权重以及四项评价指标。

- [ ] **Step 3: 注释结果保存与主流程模块**

在预测格式化、混淆矩阵、指标表、特征重要性、指标对比、`run_all()` 和 `main()` 中解释输出含义、三种实验场景、公平比较方法和命令行入口。

### Task 3: 验证行为没有改变

**Files:**
- Inspect: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Create: `D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_after_comments.txt`

**Interfaces:**
- Consumes: 完成注释的脚本和 Task 1 的基线输出。
- Produces: 语法检查与终端输出一致性结论。

- [ ] **Step 1: 再次运行语法检查**

```powershell
python -m py_compile D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py
```

Expected: 退出码为 0。

- [ ] **Step 2: 再次执行完整实验**

```powershell
python D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py | Out-File -Encoding utf8 D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_after_comments.txt
```

Expected: 正常生成七张 PNG 和 `model_metrics.csv`。

- [ ] **Step 3: 比较修改前后输出**

```powershell
Compare-Object (Get-Content D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_before_comments.txt) (Get-Content D:\UniFiles\Python_ML\.codex_tmp\bank_marketing_after_comments.txt)
```

Expected: 没有差异输出，表示注释没有改变数据量、指标和预测结果。
