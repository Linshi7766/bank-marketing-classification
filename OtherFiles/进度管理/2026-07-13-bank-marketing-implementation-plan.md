# Bank Marketing 分类结课实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一套可一键运行、结果稳定、初学者能够现场解释的 Bank Marketing 二分类实验，并生成报告所需图表、指标和前 5 条预测结果。

**Architecture:** 使用单个分段清晰的 `bank_marketing_experiment.py` 作为现场运行入口，将数据读取、质量检查、可视化、预处理、建模和评价拆成可独立测试的函数。使用 sklearn `ColumnTransformer` 与 `Pipeline` 防止预处理步骤混乱；主实验删除 `duration`，另用随机森林执行保留 `duration` 的对照实验。

**Tech Stack:** Python 3.13、pandas 3.0、NumPy 2.4、matplotlib 3.10、seaborn 0.13、scikit-learn 1.8、标准库 unittest。

## Global Constraints

- 数据文件固定为 `D:\UniFiles\Python_ML\Dataset\bank+marketing\bank\bank-full.csv`，使用 `sep=";"` 读取。
- 不修改原始 CSV。
- 运行入口固定为 `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`。
- 生成文件固定保存到 `D:\UniFiles\Python_ML\OtherFiles\bank_marketing_outputs\`。
- 固定 `random_state=42`，训练测试比例为 80%/20%，并使用 `stratify=y`。
- 主实验删除 `duration`；保留 `duration` 仅用于信息泄漏对照。
- 模型仅使用逻辑回归与随机森林，不加入复杂调参、神经网络或 SHAP。
- 最终提交报告文字由用户本人理解后撰写；实现只提供可复现结果、图表和学习辅助。
- 工作区不是 Git 仓库；计划中的“提交检查点”改为更新 `progress.md`，不初始化 Git。

## File Structure

- Create: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py` — 实验函数与 CLI 入口。
- Create: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py` — 数据清洗、特征准备、模型评价和输出格式测试。
- Create at runtime: `D:\UniFiles\Python_ML\OtherFiles\bank_marketing_outputs\*.png` — 4 张数据图、混淆矩阵和特征重要性图。
- Create at runtime: `D:\UniFiles\Python_ML\OtherFiles\bank_marketing_outputs\model_metrics.csv` — 模型指标对照表。
- Create: `D:\UniFiles\Python_ML\OtherFiles\进度管理\答辩提纲.md` — 非提交用学习提纲。
- Modify throughout: `task_plan.md`、`findings.md`、`progress.md`。

---

### Task 1: 数据读取、校验与清洗

**Files:**
- Create: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Create: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Produces: `load_dataset(path: Path) -> pd.DataFrame`
- Produces: `profile_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]`

- [x] **Step 1: 写失败测试，覆盖分号读取、目标校验、空值与重复删除**

```python
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CodeFile.bank_marketing_experiment import load_dataset, profile_and_clean


class DataQualityTests(unittest.TestCase):
    def test_load_dataset_reads_semicolon_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("age;job;y\n20;student;yes\n", encoding="utf-8")
            df = load_dataset(path)
            self.assertEqual(df.shape, (1, 3))
            self.assertEqual(df.columns.tolist(), ["age", "job", "y"])

    def test_load_dataset_rejects_missing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("age;job\n20;student\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "目标列 y"):
                load_dataset(path)

    def test_profile_and_clean_counts_then_removes_bad_rows(self):
        df = pd.DataFrame({
            "age": [20, 20, None, 30],
            "job": ["student", "student", "admin", "unknown"],
            "y": ["yes", "yes", "no", "no"],
        })
        cleaned, report = profile_and_clean(df)
        self.assertEqual(report["rows_before"], 4)
        self.assertEqual(report["missing_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["unknown_counts"], {"job": 1})
        self.assertEqual(len(cleaned), 2)
```

- [x] **Step 2: 运行测试并确认因模块或函数不存在而失败**

Run:

```powershell
py -3.13 -m unittest tests.test_bank_marketing_experiment.DataQualityTests -v
```

Expected: FAIL，提示无法导入 `bank_marketing_experiment` 或目标函数不存在。

- [x] **Step 3: 实现最小的数据读取与清洗函数**

```python
from pathlib import Path
import pandas as pd

TARGET_COLUMN = "y"


def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    df = pd.read_csv(path, sep=";")
    if TARGET_COLUMN not in df.columns:
        raise ValueError("数据中缺少目标列 y")
    invalid = set(df[TARGET_COLUMN].dropna().unique()) - {"yes", "no"}
    if invalid:
        raise ValueError(f"目标列 y 包含未知类别: {sorted(invalid)}")
    return df


def profile_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    unknown_counts = {
        col: int(df[col].astype("string").str.strip().str.lower().eq("unknown").sum())
        for col in text_columns
    }
    report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rows": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unknown_counts": {k: v for k, v in unknown_counts.items() if v > 0},
    }
    cleaned = df.dropna().drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(cleaned)
    return cleaned, report
```

- [x] **Step 4: 重跑测试并确认通过**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.DataQualityTests -v`

Expected: 3 tests PASS。

- [x] **Step 5: 更新进度文件检查点**

在 `progress.md` 记录 Task 1 的测试命令、3 项通过结果和创建的两个文件。

---

### Task 2: 四张数据可视化

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Modify: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Consumes: 清洗后的 Bank Marketing DataFrame。
- Produces: `create_visualizations(df: pd.DataFrame, output_dir: Path) -> list[Path]`

- [x] **Step 1: 写失败测试，要求生成四个非空 PNG**

```python
from CodeFile.bank_marketing_experiment import create_visualizations


class VisualizationTests(unittest.TestCase):
    def test_create_visualizations_writes_four_pngs(self):
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45],
            "job": ["student", "student", "admin", "admin", "services", "services"],
            "duration": [30, 200, 60, 400, 90, 500],
            "y": ["no", "yes", "no", "yes", "no", "yes"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            paths = create_visualizations(df, Path(tmp))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in paths))
```

- [x] **Step 2: 运行并确认测试失败**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.VisualizationTests -v`

Expected: FAIL，`create_visualizations` 不存在。

- [x] **Step 3: 实现统一绘图配置和四张图**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _save_current_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="y", hue="y", legend=False, palette="Set2")
    plt.title("定期存款购买结果数量")
    plt.xlabel("是否购买")
    plt.ylabel("记录数量")
    paths.append(_save_current_figure(output_dir / "01_target_distribution.png"))

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="y", bins=30, stat="density",
                 common_norm=False, element="step")
    plt.title("不同购买结果的年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("密度")
    paths.append(_save_current_figure(output_dir / "02_age_distribution.png"))

    job_rates = (df.assign(target=df["y"].eq("yes").astype(int))
                   .groupby("job", as_index=False)["target"].mean()
                   .sort_values("target"))
    plt.figure(figsize=(9, 6))
    sns.barplot(data=job_rates, x="target", y="job", color="#4C72B0")
    plt.title("不同职业客户的定期存款购买率")
    plt.xlabel("购买率")
    plt.ylabel("职业")
    paths.append(_save_current_figure(output_dir / "03_job_purchase_rate.png"))

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="y", y="duration", hue="y", legend=False,
                showfliers=False, palette="Set2")
    plt.title("通话时长与购买结果")
    plt.xlabel("是否购买")
    plt.ylabel("通话时长（秒）")
    paths.append(_save_current_figure(output_dir / "04_duration_boxplot.png"))
    return paths
```

- [x] **Step 4: 重跑测试并确认四张图片生成**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.VisualizationTests -v`

Expected: 1 test PASS，临时目录内 4 个 PNG 均非空。

- [x] **Step 5: 更新进度文件检查点**

记录 4 张图的固定文件名和测试通过结果。

---

### Task 3: 特征准备与模型管道

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Modify: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Produces: `prepare_xy(df, include_duration) -> tuple[pd.DataFrame, pd.Series]`
- Produces: `build_model_pipeline(X, model_name) -> sklearn.pipeline.Pipeline`

- [x] **Step 1: 写失败测试，验证标签映射、duration 开关与模型名称**

```python
from CodeFile.bank_marketing_experiment import prepare_xy, build_model_pipeline


class ModelingSetupTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "job": ["student", "admin", "services", "retired"],
            "duration": [20, 30, 40, 50],
            "y": ["no", "yes", "no", "yes"],
        })

    def test_prepare_xy_excludes_duration_for_main_model(self):
        X, y = prepare_xy(self.df, include_duration=False)
        self.assertNotIn("duration", X.columns)
        self.assertEqual(y.tolist(), [0, 1, 0, 1])

    def test_prepare_xy_can_include_duration_for_comparison(self):
        X, _ = prepare_xy(self.df, include_duration=True)
        self.assertIn("duration", X.columns)

    def test_unknown_model_name_is_rejected(self):
        X, _ = prepare_xy(self.df, include_duration=False)
        with self.assertRaisesRegex(ValueError, "未知模型"):
            build_model_pipeline(X, "svm")
```

- [x] **Step 2: 运行并确认失败**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.ModelingSetupTests -v`

Expected: FAIL，目标函数不存在。

- [x] **Step 3: 实现特征准备、ColumnTransformer 和两个模型**

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_xy(df: pd.DataFrame, include_duration: bool) -> tuple[pd.DataFrame, pd.Series]:
    drop_columns = [TARGET_COLUMN]
    if not include_duration:
        drop_columns.append("duration")
    X = df.drop(columns=drop_columns).copy()
    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("目标列 y 无法完整映射为 0/1")
    return X, y.astype(int)


def build_model_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_transformer = StandardScaler() if model_name == "logistic" else "passthrough"
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"未知模型: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
```

- [x] **Step 4: 重跑测试并确认通过**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.ModelingSetupTests -v`

Expected: 3 tests PASS。

- [x] **Step 5: 更新进度文件检查点**

记录主模型删除 `duration`、两个模型参数及建模准备测试结果。

---

### Task 4: 模型训练、评价与前 5 条预测

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Modify: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Produces: `evaluate_model(pipeline, X_train, X_test, y_train, y_test) -> dict[str, object]`
- Produces: `format_first_five_predictions(model, X_test, y_test) -> list[str]`

- [x] **Step 1: 写失败测试，验证指标键和值域及逐行输出**

```python
import numpy as np
from sklearn.dummy import DummyClassifier

from CodeFile.bank_marketing_experiment import evaluate_model, format_first_five_predictions


class EvaluationTests(unittest.TestCase):
    def test_evaluate_model_returns_required_metrics(self):
        X = pd.DataFrame({"x": list(range(10))})
        y = pd.Series([0, 1] * 5)
        model = DummyClassifier(strategy="most_frequent")
        result = evaluate_model(model, X.iloc[:8], X.iloc[8:], y.iloc[:8], y.iloc[8:])
        for key in ["accuracy", "precision_yes", "recall_yes", "f1_yes", "confusion_matrix"]:
            self.assertIn(key, result)
        self.assertTrue(0.0 <= result["accuracy"] <= 1.0)

    def test_first_five_predictions_are_one_line_each(self):
        X = pd.DataFrame({"x": list(range(8))})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        model = DummyClassifier(strategy="most_frequent").fit(X, y)
        lines = format_first_five_predictions(model, X, y)
        self.assertEqual(len(lines), 5)
        self.assertTrue(all("真实=" in line and "预测=" in line for line in lines))
```

- [x] **Step 2: 运行并确认失败**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.EvaluationTests -v`

Expected: FAIL，评价函数不存在。

- [x] **Step 3: 实现训练评价和输出格式**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(pipeline, X_train, X_test, y_train, y_test) -> dict[str, object]:
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "model": pipeline,
        "predictions": pred,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_yes": float(report["1"]["precision"]),
        "recall_yes": float(report["1"]["recall"]),
        "f1_yes": float(report["1"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, pred),
    }


def format_first_five_predictions(model, X_test: pd.DataFrame, y_test: pd.Series) -> list[str]:
    first_x = X_test.iloc[:5]
    first_y = y_test.iloc[:5].to_numpy()
    first_pred = model.predict(first_x)
    labels = {0: "no", 1: "yes"}
    return [
        f"第{i + 1}条：真实={labels[int(actual)]}，预测={labels[int(predicted)]}"
        for i, (actual, predicted) in enumerate(zip(first_y, first_pred))
    ]
```

- [x] **Step 4: 重跑测试并确认通过**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.EvaluationTests -v`

Expected: 2 tests PASS。

- [x] **Step 5: 更新进度文件检查点**

记录指标定义和“前 5 条一行一条”的测试结果。

---

### Task 5: 混淆矩阵、特征重要性与指标表

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Modify: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Produces: `save_confusion_plot(matrix, path) -> Path`
- Produces: `save_feature_importance(model_pipeline, path, top_n=10) -> Path`
- Produces: `save_metrics_table(rows, path) -> Path`

- [x] **Step 1: 写失败测试，验证模型辅助输出文件**

```python
from CodeFile.bank_marketing_experiment import save_confusion_plot, save_metrics_table


class ArtifactTests(unittest.TestCase):
    def test_confusion_and_metrics_files_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cm_path = save_confusion_plot(np.array([[8, 2], [3, 7]]), tmp / "cm.png")
            csv_path = save_metrics_table([
                {"model": "logistic", "accuracy": 0.8, "precision_yes": 0.7,
                 "recall_yes": 0.6, "f1_yes": 0.65}
            ], tmp / "metrics.csv")
            self.assertGreater(cm_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)
```

- [x] **Step 2: 运行并确认失败**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.ArtifactTests -v`

Expected: FAIL，输出函数不存在。

- [x] **Step 3: 实现混淆矩阵、指标表和随机森林特征重要性**

```python
def save_confusion_plot(matrix, path: Path) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["预测 no", "预测 yes"],
                yticklabels=["真实 no", "真实 yes"])
    plt.title("随机森林混淆矩阵（删除 duration）")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    return _save_current_figure(path)


def save_metrics_table(rows: list[dict[str, object]], path: Path) -> Path:
    columns = ["model", "accuracy", "precision_yes", "recall_yes", "f1_yes"]
    pd.DataFrame(rows)[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_feature_importance(model_pipeline: Pipeline, path: Path, top_n: int = 10) -> Path:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    forest = model_pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importance = (pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
                    .nlargest(top_n, "importance")
                    .sort_values("importance"))
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#55A868")
    plt.title(f"随机森林前 {top_n} 个重要特征")
    plt.xlabel("特征重要性")
    plt.ylabel("特征")
    return _save_current_figure(path)
```

- [x] **Step 4: 重跑测试并确认基础输出通过**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.ArtifactTests -v`

Expected: 1 test PASS。

- [x] **Step 5: 更新进度文件检查点**

记录辅助图和 CSV 文件名。

---

### Task 6: CLI 编排与完整实验运行

**Files:**
- Modify: `D:\UniFiles\Python_ML\CodeFile\bank_marketing_experiment.py`
- Modify: `D:\UniFiles\Python_ML\tests\test_bank_marketing_experiment.py`

**Interfaces:**
- Produces: `run_all(data_path: Path, output_dir: Path) -> dict[str, object]`
- Produces: `main() -> None`

- [x] **Step 1: 写失败测试，验证完整运行返回核心结果**

```python
from CodeFile.bank_marketing_experiment import run_all


class IntegrationTests(unittest.TestCase):
    def test_run_all_on_real_dataset(self):
        data_path = ROOT / "Dataset" / "bank+marketing" / "bank" / "bank-full.csv"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_all(data_path, Path(tmp))
            self.assertEqual(result["rows_before"], 45211)
            self.assertEqual(len(result["first_five"]), 5)
            self.assertIn("random_forest_without_duration", result["metrics"])
            self.assertIn("random_forest_with_duration", result["metrics"])
```

- [x] **Step 2: 运行并确认失败**

Run: `py -3.13 -m unittest tests.test_bank_marketing_experiment.IntegrationTests -v`

Expected: FAIL，`run_all` 不存在。

- [x] **Step 3: 实现完整编排**

实现顺序必须为：读取与清洗、4 张图、无 `duration` 分层划分、逻辑回归、随机森林、保留 `duration` 的随机森林对照、指标表、混淆矩阵、特征重要性、前 5 条预测。划分代码固定为：

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

对照实验必须复用相同的测试索引：先保存主实验划分产生的 `train_index` 和 `test_index`，再对包含 `duration` 的 X 使用 `.loc[train_index]` 与 `.loc[test_index]`，避免因重新随机划分造成不可比。

`run_all` 返回结构固定为：

```python
{
    "rows_before": 45211,
    "rows_after": 45211,
    "quality_report": {...},
    "metrics": {
        "logistic_without_duration": {...},
        "random_forest_without_duration": {...},
        "random_forest_with_duration": {...},
    },
    "first_five": ["第1条：真实=...，预测=...", ...],
    "artifacts": [Path(...), ...],
}
```

- [x] **Step 4: 实现 argparse 入口和清晰控制台输出**

```python
def main() -> None:
    import argparse
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bank Marketing 分类实验")
    parser.add_argument("--data", type=Path,
                        default=root / "Dataset" / "bank+marketing" / "bank" / "bank-full.csv")
    parser.add_argument("--output", type=Path,
                        default=root / "OtherFiles" / "bank_marketing_outputs")
    args = parser.parse_args()
    result = run_all(args.data, args.output)
    print(f"清洗前数据量: {result['rows_before']} 条")
    print(f"清洗后数据量: {result['rows_after']} 条")
    print("\n前5条测试数据预测结果：")
    for line in result["first_five"]:
        print(line)


if __name__ == "__main__":
    main()
```

- [x] **Step 5: 跑完整测试集**

Run:

```powershell
py -3.13 -m unittest discover -s tests -v
```

Expected: 所有测试 PASS，集成测试完成时间适合现场运行。

- [x] **Step 6: 更新进度文件检查点**

记录完整测试数量、耗时和集成测试结果。

---

### Task 7: 真实环境双运行验证与结果固化

**Files:**
- Runtime output: `D:\UniFiles\Python_ML\OtherFiles\bank_marketing_outputs\`
- Modify: `progress.md`、`findings.md`

**Interfaces:**
- Consumes: 完整实验脚本。
- Produces: 最终图表、`model_metrics.csv`、稳定控制台输出和报告数字。

- [x] **Step 1: 第一次从项目根目录运行**

Run:

```powershell
py -3.13 CodeFile\bank_marketing_experiment.py
```

Expected: exit code 0；输出清洗统计、三个模型场景指标和前 5 条预测；输出目录包含至少 7 个 PNG 与 1 个 CSV。

- [x] **Step 2: 检查输出清单和图片非空**

Run:

```powershell
Get-ChildItem -LiteralPath 'D:\UniFiles\Python_ML\OtherFiles\bank_marketing_outputs' |
  Select-Object Name,Length
```

Expected: 所有 PNG 与 CSV 的 `Length` 大于 0。

- [x] **Step 3: 第二次运行并比较核心输出**

再次运行相同命令，将控制台输出保存为临时日志并比较 Accuracy、Recall、F1 和前 5 条预测。

Expected: 两次核心数值和前 5 条预测完全一致。

- [x] **Step 4: 人工查看全部 PNG**

逐张确认：中文正常、标题清晰、坐标轴未裁剪、职业名称完整、混淆矩阵数字可读、特征重要性标签未重叠。

- [x] **Step 5: 将最终数字写入 findings.md**

仅记录最终验证运行产生的模型指标、图表结论和运行耗时，不使用中间试验数字。

- [x] **Step 6: 更新 Phase 3/4 状态**

代码和分析验证通过后，将 `task_plan.md` 的 Phase 3、Phase 4 标为 complete。

---

### Task 8: 报告支持与答辩提纲

**Files:**
- Create: `D:\UniFiles\Python_ML\OtherFiles\进度管理\答辩提纲.md`
- Modify: `progress.md`
- User-authored: `D:\UniFiles\Python_ML\实验报告\2024012046林霆恩实验报告2026.docx`

**Interfaces:**
- Consumes: 最终模型结果、图片和用户本人撰写的报告内容。
- Produces: 非提交用答辩学习提纲、报告核对清单和模拟运行记录。

- [x] **Step 1: 创建只用于学习的答辩提纲**

提纲包含 3-5 分钟时间线和以下问题的简明口语答案：

1. 为什么使用逻辑回归与随机森林？
2. 为什么不能只看准确率？
3. `class_weight="balanced"` 是什么？
4. 为什么 `unknown` 没有全部删除？
5. 为什么主模型删除 `duration`？
6. 为什么固定随机种子？
7. 数据和模型有什么局限？

- [ ] **Step 2: 用户按模板本人撰写报告**

用户使用最终图片、指标和自己的语言完成数据介绍、图表结论、关键代码说明和实验总结。助手只做事实核对、结构检查和概念讲解，不提供可直接冒充本人提交的正文。

- [x] **Step 3: 逐项核对老师要求**

核对清单：数据量、空值数、重复数、删除操作、至少 3 张图、分类模型 Accuracy、前 5 条逐行预测、源代码、运行截图、本人总结、DOCX 格式。

- [ ] **Step 4: 模拟答辩运行**

关闭旧终端，从 VS Code 新终端运行：

```powershell
py -3.13 CodeFile\bank_marketing_experiment.py
```

用户在 3-5 分钟内按提纲讲解，并能指出输出目录中的 4 张数据图、混淆矩阵和特征重要性图。

- [x] **Step 5: 更新最终进度**

在 `progress.md` 记录报告核对结果、模拟答辩问题和仍需本人复习的概念。

---

## Final Verification Gate

- [x] `py -3.13 -m unittest discover -s tests -v` 全部通过。
- [x] `py -3.13 CodeFile\bank_marketing_experiment.py` 连续两次 exit code 0。
- [x] 清洗前后数据量、空值、重复值与 `unknown` 均有输出。
- [x] 4 张数据图、主模型混淆矩阵、前 10 特征重要性均可读。
- [x] 逻辑回归、无 `duration` 随机森林、有 `duration` 随机森林指标均保存。
- [x] 前 5 条测试数据一条一行，真实值和预测值均显示。
- [ ] 报告中的数字与最终运行一致。
- [ ] 用户能够用自己的语言解释 7 个核心答辩问题。
