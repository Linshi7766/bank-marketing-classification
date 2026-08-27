import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CodeFile.bank_marketing_experiment import (
    build_model_pipeline,
    create_visualizations,
    evaluate_model,
    format_first_five_predictions,
    load_dataset,
    prepare_xy,
    profile_and_clean,
    run_all,
    save_confusion_plot,
    save_feature_importance,
    save_metrics_table,
)


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
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))

    def test_create_visualizations_has_no_deprecation_warnings(self):
        df = pd.DataFrame({
            "age": [20, 25, 30, 35],
            "job": ["student", "student", "admin", "admin"],
            "duration": [30, 200, 60, 400],
            "y": ["no", "yes", "no", "yes"],
        })
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_visualizations(df, Path(tmp))
        pending = [item for item in caught if issubclass(item.category, PendingDeprecationWarning)]
        self.assertEqual(pending, [])


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


class ArtifactTests(unittest.TestCase):
    def test_confusion_and_metrics_files_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cm_path = save_confusion_plot(np.array([[8, 2], [3, 7]]), tmp_path / "cm.png")
            csv_path = save_metrics_table([
                {
                    "model": "logistic",
                    "accuracy": 0.8,
                    "precision_yes": 0.7,
                    "recall_yes": 0.6,
                    "f1_yes": 0.65,
                }
            ], tmp_path / "metrics.csv")
            self.assertGreater(cm_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)

    def test_feature_importance_file_is_created(self):
        X = pd.DataFrame({
            "age": [20, 30, 40, 50, 60, 70],
            "job": ["student", "admin", "services", "retired", "admin", "retired"],
        })
        y = pd.Series([0, 0, 1, 1, 0, 1])
        model = build_model_pipeline(X, "random_forest").fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_feature_importance(model, Path(tmp) / "importance.png", top_n=3)
            self.assertGreater(path.stat().st_size, 0)


class IntegrationTests(unittest.TestCase):
    def test_run_all_on_real_dataset(self):
        data_path = ROOT / "Dataset" / "bank+marketing" / "bank" / "bank-full.csv"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_all(data_path, Path(tmp))
            self.assertEqual(result["rows_before"], 45211)
            self.assertEqual(len(result["first_five"]), 5)
            self.assertIn("random_forest_without_duration", result["metrics"])
            self.assertIn("random_forest_with_duration", result["metrics"])
            self.assertGreaterEqual(len(result["artifacts"]), 7)


if __name__ == "__main__":
    unittest.main()
