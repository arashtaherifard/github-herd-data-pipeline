import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "phase3_config.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_schema.json"
)

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_development.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare and tune high-growth classification "
            "models using development data only."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use reduced parameter grids for CI and "
            "pipeline smoke tests."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by GridSearchCV.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_scoring() -> dict:
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "mcc": make_scorer(
            matthews_corrcoef,
        ),
        "cohen_kappa": make_scorer(
            cohen_kappa_score,
        ),
        "neg_log_loss": "neg_log_loss",
    }


def create_model_searches(
    random_state: int,
    quick: bool,
) -> dict:
    if quick:
        return {
            "dummy": {
                "pipeline": Pipeline(
                    [
                        (
                            "model",
                            DummyClassifier(
                                random_state=random_state
                            ),
                        )
                    ]
                ),
                "grid": {
                    "model__strategy": [
                        "most_frequent",
                        "prior",
                    ]
                },
            },
            "logistic_regression": {
                "pipeline": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                solver="liblinear",
                                max_iter=5000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
                "grid": {
                    "model__C": [0.1, 1.0],
                    "model__penalty": ["l1", "l2"],
                },
            },
            "random_forest": {
                "pipeline": Pipeline(
                    [
                        (
                            "model",
                            RandomForestClassifier(
                                random_state=random_state,
                                n_jobs=1,
                            ),
                        )
                    ]
                ),
                "grid": {
                    "model__n_estimators": [100],
                    "model__max_depth": [3, None],
                    "model__min_samples_leaf": [1, 3],
                    "model__max_features": ["sqrt"],
                },
            },
            "xgboost": {
                "pipeline": Pipeline(
                    [
                        (
                            "model",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                random_state=random_state,
                                n_jobs=1,
                                verbosity=0,
                            ),
                        )
                    ]
                ),
                "grid": {
                    "model__n_estimators": [100],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                },
            },
        }

    return {
        "dummy": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        DummyClassifier(
                            random_state=random_state
                        ),
                    )
                ]
            ),
            "grid": {
                "model__strategy": [
                    "most_frequent",
                    "prior",
                    "stratified",
                ]
            },
        },
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            max_iter=5000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "grid": {
                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
                "model__penalty": ["l1", "l2"],
            },
        },
        "support_vector_machine": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "grid": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1.0, 10.0],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1.0, 10.0],
                    "model__gamma": ["scale", 0.1],
                },
            ],
        },
        "decision_tree": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeClassifier(
                            random_state=random_state
                        ),
                    )
                ]
            ),
            "grid": {
                "model__max_depth": [2, 3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__criterion": [
                    "gini",
                    "entropy",
                ],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            "grid": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": [
                    "sqrt",
                    0.7,
                ],
            },
        },
        "knn": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KNeighborsClassifier(),
                    ),
                ]
            ),
            "grid": {
                "model__n_neighbors": [3, 5, 9, 15],
                "model__weights": [
                    "uniform",
                    "distance",
                ],
                "model__p": [1, 2],
            },
        },
        "gradient_boosting": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        GradientBoostingClassifier(
                            random_state=random_state
                        ),
                    )
                ]
            ),
            "grid": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
                "model__subsample": [0.8, 1.0],
            },
        },
        "xgboost": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=random_state,
                            n_jobs=1,
                            verbosity=0,
                        ),
                    )
                ]
            ),
            "grid": {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [1.0, 5.0],
            },
        },
    }


def extract_best_result(
    model_name: str,
    search: GridSearchCV,
    elapsed_seconds: float,
) -> dict:
    results = search.cv_results_
    best_index = search.best_index_

    return {
        "model": model_name,
        "best_parameters": json.dumps(
            search.best_params_,
            sort_keys=True,
            default=str,
        ),
        "mean_cv_roc_auc": float(
            results["mean_test_roc_auc"][best_index]
        ),
        "std_cv_roc_auc": float(
            results["std_test_roc_auc"][best_index]
        ),
        "mean_cv_pr_auc": float(
            results["mean_test_pr_auc"][best_index]
        ),
        "mean_cv_accuracy": float(
            results["mean_test_accuracy"][best_index]
        ),
        "mean_cv_balanced_accuracy": float(
            results[
                "mean_test_balanced_accuracy"
            ][best_index]
        ),
        "mean_cv_precision": float(
            results["mean_test_precision"][best_index]
        ),
        "mean_cv_recall": float(
            results["mean_test_recall"][best_index]
        ),
        "mean_cv_f1": float(
            results["mean_test_f1"][best_index]
        ),
        "mean_cv_mcc": float(
            results["mean_test_mcc"][best_index]
        ),
        "mean_cv_cohen_kappa": float(
            results[
                "mean_test_cohen_kappa"
            ][best_index]
        ),
        "mean_cv_log_loss": float(
            -results[
                "mean_test_neg_log_loss"
            ][best_index]
        ),
        "mean_fit_time_seconds": float(
            results["mean_fit_time"][best_index]
        ),
        "total_search_time_seconds": (
            elapsed_seconds
        ),
        "parameter_combinations": int(
            len(results["params"])
        ),
    }


def extract_grid_results(
    model_name: str,
    search: GridSearchCV,
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)

    results.insert(0, "model", model_name)

    results["parameters_json"] = results[
        "params"
    ].apply(
        lambda value: json.dumps(
            value,
            sort_keys=True,
            default=str,
        )
    )

    desired_columns = [
        "model",
        "parameters_json",
        "rank_test_roc_auc",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_test_pr_auc",
        "std_test_pr_auc",
        "mean_test_accuracy",
        "mean_test_balanced_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_test_mcc",
        "mean_test_cohen_kappa",
        "mean_test_neg_log_loss",
        "mean_fit_time",
        "std_fit_time",
    ]

    parameter_columns = [
        column
        for column in results.columns
        if column.startswith("param_")
    ]

    selected_columns = (
        desired_columns + parameter_columns
    )

    return results[selected_columns].copy()


def create_model_comparison_plot(
    comparison: pd.DataFrame,
) -> None:
    plot_data = comparison.sort_values(
        "mean_cv_roc_auc",
        ascending=True,
    ).reset_index(drop=True)

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        plot_data["mean_cv_roc_auc"],
        positions,
        xerr=plot_data["std_cv_roc_auc"],
        fmt="o",
        capsize=4,
    )

    plt.yticks(
        positions,
        plot_data["model"],
    )

    plt.xlabel("Cross-validated ROC-AUC")
    plt.ylabel("Model")
    plt.title(
        "High-Growth Classification Model Comparison"
    )
    plt.xlim(0.0, 1.05)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "cv_roc_auc_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def create_secondary_metrics_plot(
    comparison: pd.DataFrame,
) -> None:
    plot_data = comparison.sort_values(
        "mean_cv_roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    positions = np.arange(len(plot_data))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        positions - width,
        plot_data["mean_cv_f1"],
        width=width,
        label="F1",
    )

    plt.bar(
        positions,
        plot_data["mean_cv_mcc"],
        width=width,
        label="MCC",
    )

    plt.bar(
        positions + width,
        plot_data["mean_cv_balanced_accuracy"],
        width=width,
        label="Balanced accuracy",
    )

    plt.xticks(
        positions,
        plot_data["model"],
        rotation=35,
        ha="right",
    )

    plt.ylabel("Cross-validated score")
    plt.title(
        "Secondary Classification Metrics"
    )
    plt.ylim(-0.1, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "cv_secondary_metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    arguments = parse_arguments()

    config = load_json(CONFIG_PATH)
    schema = load_json(SCHEMA_PATH)

    random_state = int(config["random_state"])
    target = schema["target"]
    feature_columns = schema["feature_columns"]

    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            "Development data was not found. Run "
            "'python scripts/build_phase3_datasets.py' "
            "first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "The untouched test file is missing. Run "
            "'python scripts/build_phase3_datasets.py' "
            "first."
        )

    development = pd.read_csv(DEVELOPMENT_PATH)

    missing_features = [
        column
        for column in feature_columns
        if column not in development.columns
    ]

    if missing_features:
        raise ValueError(
            "Development data is missing required "
            f"features: {missing_features}"
        )

    X_development = development[
        feature_columns
    ].copy()

    y_development = development[target].copy()

    if X_development.isna().any().any():
        raise ValueError(
            "Development predictors contain missing "
            "values."
        )

    cv = StratifiedKFold(
        n_splits=int(
            config["classification"][
                "cross_validation_folds"
            ]
        ),
        shuffle=True,
        random_state=random_state,
    )

    scoring = create_scoring()

    model_searches = create_model_searches(
        random_state=random_state,
        quick=arguments.quick,
    )

    comparison_rows = []
    all_grid_results = []
    fitted_searches = {}

    print("=" * 100)
    print("HIGH-GROWTH CLASSIFICATION MODEL SEARCH")
    print("=" * 100)
    print(f"Development rows: {len(development)}")
    print(f"Predictors: {len(feature_columns)}")
    print(
        "Class counts:",
        y_development.value_counts()
        .sort_index()
        .to_dict(),
    )
    print(f"Cross-validation folds: {cv.n_splits}")
    print(f"Quick mode: {arguments.quick}")
    print(
        "Untouched test rows: 29 "
        "(test file intentionally not loaded)"
    )
    print("=" * 100)

    for model_name, model_definition in (
        model_searches.items()
    ):
        print(f"\nTraining: {model_name}")

        search = GridSearchCV(
            estimator=clone(
                model_definition["pipeline"]
            ),
            param_grid=model_definition["grid"],
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=arguments.n_jobs,
            return_train_score=True,
            error_score="raise",
            verbose=0,
        )

        start_time = time.perf_counter()

        search.fit(
            X_development,
            y_development,
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        fitted_searches[model_name] = search

        comparison_row = extract_best_result(
            model_name=model_name,
            search=search,
            elapsed_seconds=elapsed_seconds,
        )

        comparison_rows.append(comparison_row)

        grid_results = extract_grid_results(
            model_name=model_name,
            search=search,
        )

        all_grid_results.append(grid_results)

        print(
            f"  Best ROC-AUC: "
            f"{comparison_row['mean_cv_roc_auc']:.4f} "
            f"+/- "
            f"{comparison_row['std_cv_roc_auc']:.4f}"
        )

        print(
            f"  Best F1: "
            f"{comparison_row['mean_cv_f1']:.4f}"
        )

        print(
            f"  Best MCC: "
            f"{comparison_row['mean_cv_mcc']:.4f}"
        )

        print(
            f"  Search time: "
            f"{elapsed_seconds:.2f} seconds"
        )

        print(
            f"  Best parameters: "
            f"{search.best_params_}"
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        [
            "mean_cv_roc_auc",
            "mean_cv_pr_auc",
            "mean_cv_mcc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "selection_rank",
        np.arange(1, len(comparison) + 1),
    )

    grid_results = pd.concat(
        all_grid_results,
        ignore_index=True,
    )

    comparison_path = (
        OUTPUT_DIR / "cv_model_comparison.csv"
    )

    grid_results_path = (
        OUTPUT_DIR / "all_hyperparameter_results.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    grid_results.to_csv(
        grid_results_path,
        index=False,
    )

    best_model_name = comparison.loc[0, "model"]

    best_search = fitted_searches[best_model_name]
    best_model = best_search.best_estimator_

    best_model_path = (
        MODEL_DIR
        / "selected_classifier_development.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    best_metadata = {
        "selected_model": best_model_name,
        "selection_metric": "mean_cv_roc_auc",
        "selection_score": float(
            comparison.loc[0, "mean_cv_roc_auc"]
        ),
        "selection_score_standard_deviation": float(
            comparison.loc[0, "std_cv_roc_auc"]
        ),
        "best_parameters": (
            best_search.best_params_
        ),
        "development_rows": int(len(development)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "random_state": random_state,
        "cross_validation": {
            "type": "StratifiedKFold",
            "folds": cv.n_splits,
            "shuffle": True,
        },
        "test_set_status": (
            "untouched_not_loaded_or_evaluated"
        ),
        "quick_mode": arguments.quick,
    }

    metadata_path = (
        MODEL_DIR
        / "selected_classifier_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            best_metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    create_model_comparison_plot(comparison)
    create_secondary_metrics_plot(comparison)

    report_lines = [
        "=" * 100,
        "HIGH-GROWTH CLASSIFICATION MODEL SEARCH",
        "=" * 100,
        "",
        f"Development rows: {len(development)}",
        f"Feature count: {len(feature_columns)}",
        f"Target: {target}",
        (
            "Test set status: untouched and not "
            "evaluated"
        ),
        "",
        "MODEL COMPARISON",
        comparison.to_string(index=False),
        "",
        f"Selected model: {best_model_name}",
        (
            "Selected cross-validated ROC-AUC: "
            f"{comparison.loc[0, 'mean_cv_roc_auc']:.6f}"
        ),
        (
            "Selected cross-validated ROC-AUC SD: "
            f"{comparison.loc[0, 'std_cv_roc_auc']:.6f}"
        ),
        (
            "Selected parameters: "
            f"{best_search.best_params_}"
        ),
        "",
        f"Saved model: {best_model_path}",
        f"Saved metadata: {metadata_path}",
        f"Saved comparison: {comparison_path}",
        f"Saved full grid results: {grid_results_path}",
    ]

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "classification_model_search_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)
    print("\nModel search completed successfully.")


if __name__ == "__main__":
    main()
