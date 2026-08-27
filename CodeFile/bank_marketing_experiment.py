"""Bank Marketing 银行营销二分类课程实验。

实验目标：根据客户资料和营销联系信息，预测客户是否会购买定期存款。
程序会依次完成数据读取、质量检查、探索性可视化、特征预处理、
模型训练与评价，并把图表和指标表保存到指定的结果文件夹。
"""

# Path 比直接拼接字符串更适合跨平台处理文件路径。
from pathlib import Path
import warnings

import matplotlib
# Agg 是非交互式绘图后端：运行程序时不弹出窗口，图片直接保存到硬盘。
# 该设置必须位于导入 pyplot 之前。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# scikit-learn 中的工具分别用于：组合预处理、建立模型、计算指标和划分数据。
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 设置中文字体候选列表。系统会从左到右选择已安装的字体。
# axes.unicode_minus=False 用于避免坐标轴负号显示为方框。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# y 是数据集中的预测目标列：yes 表示购买，no 表示不购买。
TARGET_COLUMN = "y"


def load_dataset(path: Path) -> pd.DataFrame:
    """读取原始数据，并确认目标列满足二分类实验要求。

    参数：
        path: bank-full.csv 的路径。
    返回：
        未清洗的 pandas DataFrame。

    Bank Marketing 的 CSV 使用分号而不是逗号分隔，因此读取时必须设置
    sep=";"。提前检查文件、目标列和类别值，可以让路径或数据有误时尽早报错。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_csv(path, sep=";")
    if TARGET_COLUMN not in df.columns:
        raise ValueError("数据中缺少目标列 y")

    # dropna() 只用于检查非空目标；set 差集可以找出 yes/no 之外的异常类别。
    invalid = set(df[TARGET_COLUMN].dropna().unique()) - {"yes", "no"}
    if invalid:
        raise ValueError(f"目标列 y 包含未知类别: {sorted(invalid)}")
    return df


def profile_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """统计数据质量，并执行本实验要求的基础清洗。

    返回两个对象：清洗后的 DataFrame，以及记录清洗前后情况的字典 report。
    数据集中的字符串 "unknown" 是数据提供方定义的未知类别，不是 pandas 的
    NaN 空值。这里对它单独计数并保留，避免随意删除大量有效记录。
    """
    # 只在文本列中检查 unknown，数值列不存在这种字符串类别。
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    unknown_counts = {
        col: int(df[col].astype("string").str.strip().str.lower().eq("unknown").sum())
        for col in text_columns
    }

    # 在删除记录之前保存质量统计，便于在终端和报告中说明清洗依据。
    report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rows": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unknown_counts": {key: value for key, value in unknown_counts.items() if value > 0},
    }

    # dropna 删除含真实空值的行；drop_duplicates 删除整行完全相同的记录。
    # reset_index 让删除记录后的行号重新从 0 连续排列。
    cleaned = df.dropna().drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(cleaned)
    return cleaned, report


def _save_current_figure(path: Path) -> Path:
    """统一保存当前图片，并关闭图形释放内存。

    函数名前的下划线表示它是供本脚本内部调用的辅助函数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tight_layout 和 bbox_inches="tight" 可减少标题或坐标标签被裁切的风险。
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """生成四张探索性分析图，并返回已保存图片的路径列表。

    这些图用于建模前理解数据：类别是否平衡、年龄是否存在分布差异、
    不同职业的购买率是否不同，以及通话时长与购买结果是否明显相关。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []

    # 图1：统计 yes/no 的记录数量，用于判断目标类别是否不平衡。
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="y", hue="y", legend=False, palette="Set2")
    plt.title("定期存款购买结果数量")
    plt.xlabel("是否购买")
    plt.ylabel("记录数量")
    paths.append(_save_current_figure(output_dir / "01_target_distribution.png"))

    # 图2：分别绘制购买与未购买客户的年龄密度，common_norm=False 表示
    # 两组各自归一化，便于比较分布形状，而不是比较样本绝对数量。
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="age",
        hue="y",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("不同购买结果的年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("密度")
    paths.append(_save_current_figure(output_dir / "02_age_distribution.png"))

    # 图3：把 yes 转为 1、no 转为 0；分组后的平均值就是各职业购买率。
    job_rates = (
        df.assign(target=df["y"].eq("yes").astype(int))
        .groupby("job", as_index=False)["target"]
        .mean()
        .sort_values("target")
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(data=job_rates, x="target", y="job", color="#4C72B0")
    plt.title("不同职业客户的定期存款购买率")
    plt.xlabel("购买率")
    plt.ylabel("职业")
    paths.append(_save_current_figure(output_dir / "03_job_purchase_rate.png"))

    # 图4：箱线图比较两类客户的通话时长中位数和主要分布范围。
    # showfliers=False 只隐藏极端离群点的绘制，不会删除原始数据。
    plt.figure(figsize=(7, 5))
    with warnings.catch_warnings():
        # 仅屏蔽 seaborn 内部的待弃用警告，避免终端截图出现无关提示。
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            module=r"seaborn\.categorical",
        )
        sns.boxplot(
            data=df,
            x="y",
            y="duration",
            hue="y",
            legend=False,
            showfliers=False,
            palette="Set2",
        )
    plt.title("通话时长与购买结果")
    plt.xlabel("是否购买")
    plt.ylabel("通话时长（秒）")
    paths.append(_save_current_figure(output_dir / "04_duration_boxplot.png"))
    return paths


def prepare_xy(
    df: pd.DataFrame, include_duration: bool
) -> tuple[pd.DataFrame, pd.Series]:
    """分离特征 X 和目标 y，并决定是否保留通话时长。

    X 是用于预测的客户特征，y 是模型需要学习的正确答案。
    duration 只有在营销电话结束后才能确定。如果任务是在联系客户前预测，
    使用它会把未来信息提前交给模型，形成信息泄漏。因此主模型删除该字段；
    保留 duration 的场景只用于对比信息泄漏对指标的影响。
    """
    drop_columns = [TARGET_COLUMN]
    if not include_duration:
        drop_columns.append("duration")

    X = df.drop(columns=drop_columns).copy()
    # scikit-learn 分类器使用数字标签，因此把 no/yes 映射为 0/1。
    # 本实验规定 yes=1，即后续重点评价的“正类”。
    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("目标列 y 无法完整映射为 0/1")
    return X, y.astype(int)


def build_model_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    """根据模型名称，创建“特征预处理 + 分类器”的完整 Pipeline。

    Pipeline 的好处是训练和预测会自动执行完全相同的预处理，可减少手工编码
    导致的数据不一致，也可以避免先用完整数据拟合预处理器造成信息泄漏。
    """
    # 类别特征如 job、education 是文本；其余列视为数值特征。
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    # 逻辑回归对不同数值量纲较敏感，因此使用 StandardScaler 标准化；
    # 随机森林按特征阈值分裂，对缩放不敏感，所以数值列直接传递。
    numeric_transformer = StandardScaler() if model_name == "logistic" else "passthrough"
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric),
        # 独热编码把每个文本类别转成 0/1 特征。
        # handle_unknown="ignore" 可避免测试集出现训练时未见类别而报错。
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    if model_name == "logistic":
        # 逻辑回归是结构较简单、解释性较强的基线模型。
        # balanced 会根据类别数量自动提高少数类 yes 的训练权重。
        model = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "random_forest":
        # 随机森林通过多棵决策树投票完成分类，可以学习非线性关系。
        # n_estimators 是树的数量；max_depth 和 min_samples_leaf 用于限制复杂度；
        # n_jobs=-1 表示使用所有可用 CPU 核心；random_state 保证结果可复现。
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

    # 训练时先拟合 preprocessor，再把转换后的特征交给 model。
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_model(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """训练模型，在测试集上预测，并计算 yes 类别的主要指标。

    数据中的 no 明显多于 yes，因此 Accuracy 较高不一定代表模型能找出购买者。
    本实验还计算 yes 类别的 Precision、Recall 和 F1：
    - Precision：预测会购买的客户中，实际购买者所占比例；
    - Recall：实际购买者中，被模型成功找到的比例；
    - F1：Precision 与 Recall 的调和平均，用于综合评价二者。
    """
    # fit 只接触训练集；训练完成后再对独立测试集进行预测。
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # output_dict=True 让评价报告以字典返回，便于提取类别 1（yes）的指标。
    # zero_division=0 用于处理某类没有任何预测结果时的除零问题。
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    return {
        # 保存已训练模型与预测值，后续绘图和展示前五条预测时继续使用。
        "model": pipeline,
        "predictions": predictions,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_yes": float(report["1"]["precision"]),
        "recall_yes": float(report["1"]["recall"]),
        "f1_yes": float(report["1"]["f1-score"]),
        "classification_report": report,
        # labels=[0, 1] 固定矩阵的行列顺序，避免不同运行时含义混乱。
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }


def format_first_five_predictions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[str]:
    """比较测试集前五条记录的真实类别和预测类别，生成可读文本。"""
    # iloc[:5] 按测试集当前顺序取前五条，并不会重新选择或修改测试数据。
    first_x = X_test.iloc[:5]
    first_y = y_test.iloc[:5].to_numpy()
    first_predictions = model.predict(first_x)
    # 将模型使用的 0/1 恢复为报告中更直观的 no/yes。
    labels = {0: "no", 1: "yes"}
    return [
        f"第{i + 1}条：真实={labels[int(actual)]}，预测={labels[int(predicted)]}"
        for i, (actual, predicted) in enumerate(zip(first_y, first_predictions))
    ]


def save_confusion_plot(matrix, path: Path) -> Path:
    """把主模型的混淆矩阵保存为热力图。

    纵轴是真实类别，横轴是预测类别；主对角线表示预测正确的数量。
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["预测 no", "预测 yes"],
        yticklabels=["真实 no", "真实 yes"],
    )
    plt.title("随机森林混淆矩阵（删除 duration）")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    return _save_current_figure(Path(path))


def save_metrics_table(rows: list[dict[str, object]], path: Path) -> Path:
    """把三个实验场景的主要指标保存为 CSV，便于检查和引用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "accuracy", "precision_yes", "recall_yes", "f1_yes"]
    pd.DataFrame(rows)[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_feature_importance(
    model_pipeline: Pipeline,
    path: Path,
    top_n: int = 10,
) -> Path:
    """提取并绘制随机森林中重要性最高的若干个编码后特征。

    特征重要性表示该特征对随机森林分裂的贡献程度，但只能说明模型中的
    预测作用，不能直接证明该特征与购买行为之间存在因果关系。
    """
    # 通过 Pipeline 的步骤名称分别取得已拟合的预处理器和随机森林。
    preprocessor = model_pipeline.named_steps["preprocessor"]
    forest = model_pipeline.named_steps["model"]

    # 独热编码会扩展类别列，因此必须从预处理器取得转换后的完整特征名。
    names = preprocessor.get_feature_names_out()
    importance = (
        pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
        # 先选出最大的 top_n 项，再按升序排列，便于横向柱状图从小到大展示。
        .nlargest(top_n, "importance")
        .sort_values("importance")
    )
    # 去掉 ColumnTransformer 自动添加的步骤前缀，让图片标签更简洁。
    importance["feature"] = (
        importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#55A868")
    plt.title(f"随机森林前 {top_n} 个重要特征")
    plt.xlabel("特征重要性")
    plt.ylabel("特征")
    return _save_current_figure(Path(path))


def _save_metrics_comparison(rows: list[dict[str, object]], path: Path) -> Path:
    """把三个实验场景的四项指标绘制成分组柱状图。"""
    # 程序内部使用稳定的英文名称，绘图时再映射为便于报告阅读的中文名称。
    display_names = {
        "logistic_without_duration": "逻辑回归\n无 duration",
        "random_forest_without_duration": "随机森林\n无 duration",
        "random_forest_with_duration": "随机森林\n有 duration",
    }
    metrics = pd.DataFrame(rows).rename(columns={"model": "实验场景"})
    metrics["实验场景"] = metrics["实验场景"].map(display_names)

    # melt 将“每项指标一列”的宽表转换成长表，以便 seaborn 按指标分组绘图。
    long_metrics = metrics.melt(
        id_vars="实验场景",
        value_vars=["accuracy", "precision_yes", "recall_yes", "f1_yes"],
        var_name="指标",
        value_name="数值",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_metrics, x="实验场景", y="数值", hue="指标")
    plt.ylim(0, 1)
    plt.title("三个模型场景的评价指标对比")
    plt.xlabel("模型场景")
    plt.ylabel("指标值")
    plt.legend(title="指标", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_current_figure(Path(path))


def _metric_row(name: str, result: dict[str, object]) -> dict[str, object]:
    """从完整模型结果中提取绘图和保存 CSV 所需的五个字段。"""
    return {
        "model": name,
        "accuracy": result["accuracy"],
        "precision_yes": result["precision_yes"],
        "recall_yes": result["recall_yes"],
        "f1_yes": result["f1_yes"],
    }


def run_all(data_path: Path, output_dir: Path) -> dict[str, object]:
    """按顺序执行完整实验，并汇总报告与答辩需要的结果。

    本实验只有两种模型算法：逻辑回归和随机森林，但设置了三个实验场景：
    1. 逻辑回归，不使用 duration；
    2. 随机森林，不使用 duration（主模型）；
    3. 随机森林，使用 duration（信息泄漏对比场景）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 第一阶段：读取数据、统计质量并生成四张探索性分析图。
    raw_df = load_dataset(data_path)
    cleaned_df, quality_report = profile_and_clean(raw_df)
    artifacts = create_visualizations(cleaned_df, output_dir)

    # 第二阶段：主实验删除 duration，再按 8:2 划分训练集和测试集。
    X, y = prepare_xy(cleaned_df, include_duration=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        # 固定随机种子，使每次运行都得到相同划分，便于复现实验。
        random_state=42,
        # 分层抽样保证训练集和测试集中的 yes/no 比例接近原始数据。
        stratify=y,
    )

    # 保存本次划分的行索引，确保有 duration 的对比场景使用完全相同的样本。
    # 这样指标差异主要来自特征和模型，而不是来自不同的数据划分。
    train_index = X_train.index
    test_index = X_test.index

    # 场景1：不使用 duration 的逻辑回归，作为容易解释的基线模型。
    logistic = evaluate_model(
        build_model_pipeline(X, "logistic"),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 场景2：不使用 duration 的随机森林，是本实验选定的主模型。
    forest_without_duration = evaluate_model(
        build_model_pipeline(X, "random_forest"),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 场景3：重新取得包含 duration 的特征，但沿用上面的训练/测试索引。
    # 它仍然是随机森林，并不是“第三种模型”。
    X_with_duration, _ = prepare_xy(cleaned_df, include_duration=True)
    forest_with_duration = evaluate_model(
        build_model_pipeline(X_with_duration, "random_forest"),
        X_with_duration.loc[train_index],
        X_with_duration.loc[test_index],
        y.loc[train_index],
        y.loc[test_index],
    )

    # 使用统一名称组织三个场景的结果，方便终端打印、保存 CSV 和绘图。
    metrics = {
        "logistic_without_duration": logistic,
        "random_forest_without_duration": forest_without_duration,
        "random_forest_with_duration": forest_with_duration,
    }
    metric_rows = [_metric_row(name, result) for name, result in metrics.items()]

    # 第三阶段：保存指标表和三张模型分析图。
    artifacts.append(save_metrics_table(metric_rows, output_dir / "model_metrics.csv"))
    artifacts.append(save_confusion_plot(
        forest_without_duration["confusion_matrix"],
        output_dir / "05_confusion_matrix.png",
    ))
    artifacts.append(save_feature_importance(
        forest_without_duration["model"],
        output_dir / "06_feature_importance.png",
        top_n=10,
    ))
    artifacts.append(_save_metrics_comparison(
        metric_rows,
        output_dir / "07_model_metrics_comparison.png",
    ))

    # 使用主模型展示前五条测试记录的真实值和预测值。
    first_five = format_first_five_predictions(
        forest_without_duration["model"],
        X_test,
        y_test,
    )

    # 返回结构化结果，main() 只负责将其中的关键信息打印到终端。
    return {
        "rows_before": len(raw_df),
        "rows_after": len(cleaned_df),
        "quality_report": quality_report,
        "metrics": metrics,
        "first_five": first_five,
        "artifacts": artifacts,
    }


def main() -> None:
    """解析命令行参数，运行实验并打印适合截图的结果摘要。"""
    # argparse 让用户既可以使用默认路径，也可以通过 --data 和 --output
    # 指定其他数据文件或结果文件夹。
    import argparse

    # 当前文件位于项目的 CodeFile 文件夹，因此 parents[1] 是项目根目录。
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bank Marketing 分类实验")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "Dataset" / "bank+marketing" / "bank" / "bank-full.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "OtherFiles" / "bank_marketing_outputs",
    )
    args = parser.parse_args()

    # run_all 执行所有计算；从这里开始不再重复处理数据，只负责展示结果。
    result = run_all(args.data, args.output)

    quality = result["quality_report"]
    print("=" * 62)
    print("Bank Marketing 二分类实验")
    print("=" * 62)
    print(f"清洗前数据量: {result['rows_before']} 条")
    print(f"清洗后数据量: {result['rows_after']} 条")
    print(f"真实空值单元格: {quality['missing_cells']} 个")
    print(f"含空值的行: {quality['missing_rows']} 行")
    print(f"完全重复行: {quality['duplicate_rows']} 行")
    print(f"unknown 计数: {quality['unknown_counts']}")

    # yes 是本实验关注的正类，所以表中 Precision、Recall、F1 均指 yes 类别。
    print("\n模型评价指标（yes 为正类）：")
    print(f"{'模型场景':<34}{'Accuracy':>10}{'Precision':>11}{'Recall':>10}{'F1':>10}")
    for name, metrics in result["metrics"].items():
        print(
            f"{name:<34}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision_yes']:>11.4f}"
            f"{metrics['recall_yes']:>10.4f}"
            f"{metrics['f1_yes']:>10.4f}"
        )

    # 前五条预测用于满足实验要求，并让答辩时能够直观看到模型输出格式。
    print("\n前5条测试数据预测结果（主模型：无 duration 随机森林）：")
    for line in result["first_five"]:
        print(line)
    print(f"\n结果文件夹: {args.output.resolve()}")


# 只有直接运行本文件时才调用 main；作为模块被导入时不会自动执行完整实验。
if __name__ == "__main__":
    main()
