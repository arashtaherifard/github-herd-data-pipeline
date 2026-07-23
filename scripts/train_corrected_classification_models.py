import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from classification_temporal_cv import (
    build_repository_aware_temporal_cv,
    load_classification_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "modeling" / "classification"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3" / "classification" / "corrected_model_search"
MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "corrected"
CANDIDATE_MODEL_DIR = MODEL_DIR / "candidates"

for directory in [OUTPUT_DIR, MODEL_DIR, CANDIDATE_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "classification_train.csv"
VALIDATION_PATH = DATA_DIR / "classification_validation.csv"
SCHEMA_PATH = DATA_DIR / "classification_schema.json"
RANDOM_STATE = 42


def load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_features(frame: pd.DataFrame, feature_columns: list[str], name: str) -> None:
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing features: {sorted(missing)}")

    numeric = frame[feature_columns].apply(pd.to_numeric, errors="raise")

    if numeric.isna().any().any():
        raise ValueError(f"{name} contains missing feature values.")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite feature values.")


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_array, probabilities)),
        "pr_auc": float(average_precision_score(y_true_array, probabilities)),
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, predictions)
        ),
        "precision": float(
            precision_score(y_true_array, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_array, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true_array, predictions)),
        "log_loss": float(log_loss(y_true_array, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_array, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_validation_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.asarray(probabilities, dtype=float),
            ]
        )
    )

    rows = []

    for threshold in candidate_thresholds:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        rows.append(metrics)

    results = pd.DataFrame(rows)
    results["distance_from_0_5"] = (results["threshold"] - 0.5).abs()

    selected = results.sort_values(
        ["mcc", "balanced_accuracy", "f1", "distance_from_0_5"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return float(selected["threshold"]), results


def build_scoring() -> dict:
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "mcc": make_scorer(matthews_corrcoef),
        "cohen_kappa": make_scorer(cohen_kappa_score),
        "neg_log_loss": "neg_log_loss",
        "neg_brier_score": "neg_brier_score",
    }


def build_model_specs(class_ratio: float) -> dict[str, dict]:
    return {
        "dummy": {
            "estimator": Pipeline(
                [("model", DummyClassifier(random_state=RANDOM_STATE))]
            ),
            "param_grid": {"model__strategy": ["prior"]},
        },
        "logistic_regression": {
            "estimator": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            l1_ratio=0.0,
                            max_iter=10000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.0, 0.5, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "knn": {
            "estimator": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "model__n_neighbors": [3, 5, 9, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "decision_tree": {
            "estimator": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeClassifier(random_state=RANDOM_STATE),
                    )
                ]
            ),
            "param_grid": {
                "model__max_depth": [2, 4, 6, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            "param_grid": {
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": ["sqrt", 0.75],
                "model__class_weight": [None, "balanced_subsample"],
            },
        },
        "gradient_boosting": {
            "estimator": Pipeline(
                [
                    (
                        "model",
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                    )
                ]
            ),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
                "model__subsample": [0.8, 1.0],
            },
        },
        "hist_gradient_boosting": {
            "estimator": Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.03, 0.1],
                "model__max_iter": [100, 200],
                "model__max_leaf_nodes": [7, 15],
                "model__l2_regularization": [0.0, 1.0],
            },
        },
        "xgboost": {
            "estimator": Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            "param_grid": {
                "model__n_estimators": [100, 250],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__scale_pos_weight": [1.0, class_ratio],
            },
        },
    }


def extract_best_cv_metrics(search: GridSearchCV) -> dict[str, float]:
    index = int(search.best_index_)
    results = search.cv_results_

    return {
        "cv_roc_auc_mean": float(results["mean_test_roc_auc"][index]),
        "cv_roc_auc_std": float(results["std_test_roc_auc"][index]),
        "cv_pr_auc_mean": float(results["mean_test_pr_auc"][index]),
        "cv_pr_auc_std": float(results["std_test_pr_auc"][index]),
        "cv_accuracy_mean": float(results["mean_test_accuracy"][index]),
        "cv_balanced_accuracy_mean": float(
            results["mean_test_balanced_accuracy"][index]
        ),
        "cv_precision_mean": float(results["mean_test_precision"][index]),
        "cv_recall_mean": float(results["mean_test_recall"][index]),
        "cv_f1_mean": float(results["mean_test_f1"][index]),
        "cv_mcc_mean": float(results["mean_test_mcc"][index]),
        "cv_cohen_kappa_mean": float(
            results["mean_test_cohen_kappa"][index]
        ),
        "cv_log_loss_mean": float(-results["mean_test_neg_log_loss"][index]),
        "cv_brier_score_mean": float(
            -results["mean_test_neg_brier_score"][index]
        ),
    }


def save_search_results(model_name: str, search: GridSearchCV) -> pd.DataFrame:
    frame = pd.DataFrame(search.cv_results_)
    frame.insert(0, "model", model_name)
    frame["params_json"] = frame["params"].apply(
        lambda value: json.dumps(value, sort_keys=True, default=str)
    )
    return frame


def plot_validation_curves(
    validation_targets: pd.Series,
    probability_map: dict[str, np.ndarray],
) -> None:
    plt.figure(figsize=(9, 7))
    for model_name, probabilities in probability_map.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            validation_targets,
            probabilities,
        )
        auc_value = roc_auc_score(validation_targets, probabilities)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Corrected Growth-Surge Classification: Validation ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 7))
    baseline = float(validation_targets.mean())
    for model_name, probabilities in probability_map.items():
        precision, recall, _ = precision_recall_curve(
            validation_targets,
            probabilities,
        )
        auc_value = average_precision_score(validation_targets, probabilities)
        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AP={auc_value:.3f})",
        )
    plt.axhline(baseline, linestyle="--", label=f"Prevalence={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Corrected Growth-Surge Classification: Validation PR")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_precision_recall_curves.png", dpi=300)
    plt.close()


def main() -> None:
    settings = load_classification_config()
    schema = load_schema()

    target = settings["target"]
    feature_columns = list(schema["feature_columns"])

    train_raw = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    validate_features(train_raw, feature_columns, "training data")
    validate_features(validation, feature_columns, "validation data")

    prepared_train, folds, fold_summary, _ = build_repository_aware_temporal_cv(
        frame=train_raw,
        target_column=target,
        n_splits=int(settings["cross_validation_folds"]),
        validation_samples_per_repository=int(
            settings[
                "cross_validation_validation_samples_per_repository"
            ]
        ),
        gap_samples=int(settings["purge_weeks"]),
        require_all_repositories=bool(
            settings["cross_validation_require_all_repositories"]
        ),
    )

    validation["cutoff_week"] = pd.to_datetime(
        validation["cutoff_week"], utc=True, errors="raise"
    )
    validation = validation.sort_values(
        ["repo_id", "cutoff_week"]
    ).reset_index(drop=True)

    X_train = prepared_train[feature_columns]
    y_train = prepared_train[target].astype(int)
    X_validation = validation[feature_columns]
    y_validation = validation[target].astype(int)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    class_ratio = negative_count / positive_count

    specs = build_model_specs(class_ratio)
    scoring = build_scoring()

    comparison_rows = []
    all_search_results = []
    validation_prediction_rows = []
    validation_probabilities = {}
    fitted_searches = {}

    print("=" * 105)
    print("CORRECTED GROWTH-SURGE CLASSIFICATION MODEL SEARCH")
    print("=" * 105)
    print("Training rows:", len(prepared_train))
    print("Validation rows:", len(validation))
    print("Features:", len(feature_columns))
    print("Training target counts:", y_train.value_counts().sort_index().to_dict())
    print("Validation target counts:", y_validation.value_counts().sort_index().to_dict())
    print("Temporal folds:", len(folds))
    print("Test data: not loaded")

    for model_name, specification in specs.items():
        print("\n" + "-" * 105)
        print(f"Searching {model_name}...")

        search = GridSearchCV(
            estimator=specification["estimator"],
            param_grid=specification["param_grid"],
            scoring=scoring,
            refit="roc_auc",
            cv=folds,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
            verbose=0,
        )

        search.fit(X_train, y_train)
        fitted_searches[model_name] = search

        probabilities = search.best_estimator_.predict_proba(X_validation)[:, 1]
        validation_probabilities[model_name] = probabilities

        cv_metrics = extract_best_cv_metrics(search)
        validation_metrics = calculate_metrics(
            y_validation,
            probabilities,
            threshold=0.5,
        )

        row = {
            "model": model_name,
            "best_parameters": json.dumps(
                search.best_params_, sort_keys=True, default=str
            ),
            **cv_metrics,
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        comparison_rows.append(row)

        search_results = save_search_results(model_name, search)
        all_search_results.append(search_results)

        candidate_path = CANDIDATE_MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(search.best_estimator_, candidate_path)

        for row_index, (probability, truth) in enumerate(
            zip(probabilities, y_validation)
        ):
            validation_prediction_rows.append(
                {
                    "model": model_name,
                    "validation_row": row_index,
                    "repo_id": validation.loc[row_index, "repo_id"],
                    "repo_full_name": validation.loc[
                        row_index, "repo_full_name"
                    ],
                    "cutoff_week": str(validation.loc[row_index, "cutoff_week"]),
                    "target": int(truth),
                    "probability": float(probability),
                    "prediction_at_0_5": int(probability >= 0.5),
                }
            )

        print("Best parameters:", search.best_params_)
        print(
            "CV ROC-AUC:",
            f"{cv_metrics['cv_roc_auc_mean']:.4f} ± "
            f"{cv_metrics['cv_roc_auc_std']:.4f}",
        )
        print("Validation ROC-AUC:", f"{validation_metrics['roc_auc']:.4f}")
        print("Validation PR-AUC:", f"{validation_metrics['pr_auc']:.4f}")

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        [
            "validation_roc_auc",
            "validation_pr_auc",
            "cv_roc_auc_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    eligible_comparison = comparison[comparison["model"] != "dummy"]
    selected_row = eligible_comparison.iloc[0]
    selected_model_name = str(selected_row["model"])
    selected_search = fitted_searches[selected_model_name]
    selected_estimator = selected_search.best_estimator_
    selected_probabilities = validation_probabilities[selected_model_name]

    selected_threshold, threshold_results = choose_validation_threshold(
        y_validation,
        selected_probabilities,
    )
    selected_threshold_metrics = calculate_metrics(
        y_validation,
        selected_probabilities,
        selected_threshold,
    )

    ablation_feature = settings["target_feature_ablation"]["feature"]
    ablated_features = [
        feature for feature in feature_columns if feature != ablation_feature
    ]

    selected_specification = build_model_specs(class_ratio)[selected_model_name]
    ablation_search = GridSearchCV(
        estimator=selected_specification["estimator"],
        param_grid=selected_specification["param_grid"],
        scoring=scoring,
        refit="roc_auc",
        cv=folds,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
        verbose=0,
    )
    ablation_search.fit(prepared_train[ablated_features], y_train)
    ablation_probabilities = ablation_search.best_estimator_.predict_proba(
        validation[ablated_features]
    )[:, 1]
    ablation_cv_metrics = extract_best_cv_metrics(ablation_search)
    ablation_validation_metrics = calculate_metrics(
        y_validation,
        ablation_probabilities,
        threshold=0.5,
    )

    ablation_summary = pd.DataFrame(
        [
            {
                "design": "full_features",
                "model": selected_model_name,
                "feature_count": len(feature_columns),
                "removed_feature": "none",
                "cv_roc_auc": selected_row["cv_roc_auc_mean"],
                "validation_roc_auc": selected_row["validation_roc_auc"],
                "validation_pr_auc": selected_row["validation_pr_auc"],
                "validation_log_loss": selected_row["validation_log_loss"],
                "validation_brier_score": selected_row[
                    "validation_brier_score"
                ],
            },
            {
                "design": "target_feature_ablation",
                "model": selected_model_name,
                "feature_count": len(ablated_features),
                "removed_feature": ablation_feature,
                "cv_roc_auc": ablation_cv_metrics["cv_roc_auc_mean"],
                "validation_roc_auc": ablation_validation_metrics["roc_auc"],
                "validation_pr_auc": ablation_validation_metrics["pr_auc"],
                "validation_log_loss": ablation_validation_metrics[
                    "log_loss"
                ],
                "validation_brier_score": ablation_validation_metrics[
                    "brier_score"
                ],
            },
        ]
    )

    importance = permutation_importance(
        selected_estimator,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    all_search_results_frame = pd.concat(
        all_search_results,
        ignore_index=True,
    )
    validation_predictions = pd.DataFrame(validation_prediction_rows)

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    all_search_results_frame.to_csv(
        OUTPUT_DIR / "all_hyperparameter_results.csv", index=False
    )
    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions_all_models.csv", index=False
    )
    threshold_results.to_csv(
        OUTPUT_DIR / "selected_model_threshold_search.csv", index=False
    )
    fold_summary.to_csv(OUTPUT_DIR / "temporal_cv_folds_used.csv", index=False)
    ablation_summary.to_csv(OUTPUT_DIR / "target_feature_ablation.csv", index=False)
    importance_frame.to_csv(
        OUTPUT_DIR / "selected_model_permutation_importance.csv", index=False
    )

    selected_validation_frame = validation[
        ["repo_id", "repo_full_name", "cutoff_week", "target_end_week", target]
    ].copy()
    selected_validation_frame["predicted_probability"] = selected_probabilities
    selected_validation_frame["prediction_at_0_5"] = (
        selected_probabilities >= 0.5
    ).astype(int)
    selected_validation_frame["selected_threshold"] = selected_threshold
    selected_validation_frame["prediction_at_selected_threshold"] = (
        selected_probabilities >= selected_threshold
    ).astype(int)
    selected_validation_frame.to_csv(
        OUTPUT_DIR / "selected_model_validation_predictions.csv",
        index=False,
    )

    joblib.dump(selected_estimator, MODEL_DIR / "selected_classifier.joblib")
    joblib.dump(
        ablation_search.best_estimator_,
        MODEL_DIR / "selected_classifier_target_feature_ablation.joblib",
    )

    plot_validation_curves(y_validation, validation_probabilities)

    plt.figure(figsize=(9, 7))
    probability_true, probability_predicted = calibration_curve(
        y_validation,
        selected_probabilities,
        n_bins=6,
        strategy="quantile",
    )
    plt.plot(
        probability_predicted,
        probability_true,
        marker="o",
        label=selected_model_name,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Selected Classifier Validation Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_model_calibration.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 7))
    plotted_importance = importance_frame.sort_values(
        "importance_mean"
    )
    plt.barh(
        plotted_importance["feature"],
        plotted_importance["importance_mean"],
        xerr=plotted_importance["importance_std"],
    )
    plt.xlabel("Decrease in validation ROC-AUC after permutation")
    plt.ylabel("Feature")
    plt.title("Selected Classifier Permutation Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_model_permutation_importance.png", dpi=300)
    plt.close()

    metadata = {
        "design_status": "corrected_future_growth_surge_classifier",
        "target": target,
        "selected_model": selected_model_name,
        "selection_rule": (
            "Highest validation ROC-AUC, then validation PR-AUC, "
            "then training temporal-CV ROC-AUC; dummy excluded."
        ),
        "best_parameters": selected_search.best_params_,
        "feature_columns": feature_columns,
        "selected_threshold": selected_threshold,
        "selected_threshold_rule": (
            "Maximum validation MCC, then balanced accuracy, then F1."
        ),
        "validation_metrics_at_selected_threshold": selected_threshold_metrics,
        "target_feature_ablation": {
            "removed_feature": ablation_feature,
            "best_parameters": ablation_search.best_params_,
            "validation_metrics_at_0_5": ablation_validation_metrics,
        },
        "training_rows": int(len(prepared_train)),
        "validation_rows": int(len(validation)),
        "repositories": int(prepared_train["repo_id"].nunique()),
        "temporal_cv_folds": len(folds),
        "test_status": "not_loaded_or_evaluated",
        "software": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    (MODEL_DIR / "selected_classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "=" * 105,
        "CORRECTED GROWTH-SURGE CLASSIFICATION REPORT",
        "=" * 105,
        "",
        f"Training rows: {len(prepared_train)}",
        f"Validation rows: {len(validation)}",
        f"Repositories: {prepared_train['repo_id'].nunique()}",
        f"Features: {len(feature_columns)}",
        "Test data was not loaded or evaluated.",
        "",
        "MODEL COMPARISON",
        comparison.to_string(index=False),
        "",
        f"Selected model: {selected_model_name}",
        f"Selected parameters: {selected_search.best_params_}",
        f"Selected validation threshold: {selected_threshold:.6f}",
        "",
        "SELECTED-THRESHOLD VALIDATION METRICS",
        json.dumps(selected_threshold_metrics, indent=2),
        "",
        "TARGET-FEATURE ABLATION",
        ablation_summary.to_string(index=False),
        "",
        "TOP PERMUTATION IMPORTANCES",
        importance_frame.head(12).to_string(index=False),
    ]

    report = "\n".join(report_lines)
    (OUTPUT_DIR / "classification_model_report.txt").write_text(
        report + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 105)
    print("MODEL COMPARISON")
    print("=" * 105)
    print(
        comparison[
            [
                "model",
                "cv_roc_auc_mean",
                "cv_roc_auc_std",
                "validation_roc_auc",
                "validation_pr_auc",
                "validation_f1",
                "validation_mcc",
                "validation_log_loss",
                "validation_brier_score",
            ]
        ].to_string(index=False)
    )

    print("\nSelected model:", selected_model_name)
    print("Selected threshold:", f"{selected_threshold:.6f}")
    print("Validation metrics at selected threshold:")
    print(json.dumps(selected_threshold_metrics, indent=2))
    print("\nTarget-feature ablation:")
    print(ablation_summary.to_string(index=False))
    print("\nTest data was not loaded or evaluated.")
    print("\nCorrected classification model search completed successfully.")


if __name__ == "__main__":
    main()
