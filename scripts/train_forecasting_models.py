import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from forecasting_temporal_cv import build_repository_aware_forecasting_cv

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/phase3_config.json"
SCHEMA_PATH = ROOT / "data/modeling/forecasting/forecasting_schema.json"
TRAIN_PATH = ROOT / "data/modeling/forecasting/forecast_primary_train.csv"
VALIDATION_PATH = ROOT / "data/modeling/forecasting/forecast_primary_validation.csv"
OUTPUT_DIR = ROOT / "outputs/phase3/forecasting/model_search"
MODEL_DIR = ROOT / "models/forecasting"
CANDIDATE_DIR = MODEL_DIR / "candidates"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_values = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(y_true, dtype=float),
        where=denominator != 0,
    )
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(np.mean(smape_values)),
        "forecast_bias": float(np.mean(y_pred - y_true)),
    }


def validate_frame(frame: pd.DataFrame, name: str, features: list[str], target: str) -> pd.DataFrame:
    required = set(features + ["repo_id", "repo_full_name", "cutoff_week", "target_end_week", target])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["cutoff_week"] = pd.to_datetime(result["cutoff_week"], utc=True, errors="raise")
    result["target_end_week"] = pd.to_datetime(result["target_end_week"], utc=True, errors="raise")
    result[features] = result[features].apply(pd.to_numeric, errors="raise")
    result[target] = pd.to_numeric(result[target], errors="raise")

    if result[features].isna().any().any() or not np.isfinite(result[features].to_numpy(float)).all():
        raise ValueError(f"{name} contains missing or non-finite feature values.")
    if result[target].isna().any() or (result[target] < 0).any():
        raise ValueError(f"{name} contains invalid target values.")
    if result.duplicated(["repo_id", "cutoff_week"]).any():
        raise ValueError(f"{name} contains duplicate repository/cutoff keys.")
    return result


def wrap_target(estimator, transform: str):
    if transform == "raw":
        return estimator
    if transform == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    raise ValueError(f"Unknown target transform: {transform}")


def build_estimator(model: str, params: dict, transform: str):
    if model == "dummy_mean":
        estimator = DummyRegressor(strategy="mean")
    elif model == "dummy_median":
        estimator = DummyRegressor(strategy="median")
    elif model == "ridge":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(params["alpha"]))),
        ])
    elif model == "elastic_net":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]),
                max_iter=20000,
                random_state=RANDOM_STATE,
            )),
        ])
    elif model == "poisson_regression":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", PoissonRegressor(alpha=float(params["alpha"]), max_iter=5000)),
        ])
    elif model == "knn":
        estimator = Pipeline([
            ("scaler", RobustScaler()),
            ("regressor", KNeighborsRegressor(
                n_neighbors=int(params["n_neighbors"]),
                weights=params["weights"],
                p=int(params["p"]),
            )),
        ])
    elif model == "decision_tree":
        estimator = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
        )
    elif model == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            loss=params["loss"],
            random_state=RANDOM_STATE,
        )
    elif model == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            loss="squared_error",
            random_state=RANDOM_STATE,
        )
    elif model == "xgboost":
        estimator = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    return wrap_target(estimator, transform)


def search_spaces() -> dict:
    return {
        "dummy_mean": {"transforms": ["raw"], "grid": [{}]},
        "dummy_median": {"transforms": ["raw"], "grid": [{}]},
        "ridge": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]})),
        },
        "elastic_net": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.5, 0.9],
            })),
        },
        "poisson_regression": {
            "transforms": ["raw"],
            "grid": list(ParameterGrid({"alpha": [0.0, 0.01, 0.1, 1.0, 10.0]})),
        },
        "knn": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            })),
        },
        "decision_tree": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "max_depth": [2, 4, 6, None],
                "min_samples_leaf": [5, 10, 20],
                "max_features": [None, "sqrt"],
            })),
        },
        "random_forest": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "max_depth": [4, 8, None],
                "min_samples_leaf": [1, 5, 10],
                "max_features": ["sqrt", 1.0],
            })),
        },
        "extra_trees": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "max_depth": [4, 8, None],
                "min_samples_leaf": [1, 5, 10],
                "max_features": ["sqrt", 1.0],
            })),
        },
        "gradient_boosting": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [1, 2],
                "min_samples_leaf": [5, 10],
                "loss": ["squared_error", "huber"],
            })),
        },
        "hist_gradient_boosting": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "learning_rate": [0.03, 0.1],
                "max_iter": [150, 300],
                "max_leaf_nodes": [7, 15],
                "min_samples_leaf": [10, 20],
                "l2_regularization": [0.0, 1.0],
            })),
        },
        "xgboost": {
            "transforms": ["raw", "log1p"],
            "grid": list(ParameterGrid({
                "n_estimators": [150, 350],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 4],
                "min_child_weight": [1.0, 5.0],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "reg_alpha": [0.0],
                "reg_lambda": [1.0, 10.0],
            })),
        },
    }


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "last_week_x4": 4.0 * frame["weekly_new_stars"].to_numpy(float),
        "previous_week_x4": 4.0 * frame["previous_week_stars"].to_numpy(float),
        "rolling_3week_mean_x4": 4.0 * frame["rolling_3week_mean_stars"].to_numpy(float),
        "previous_4week_total": frame["rolling_4week_sum_stars"].to_numpy(float),
    }


def save_plots(comparison, selected_predictions, importance, target, model_name):
    plot_frame = comparison.sort_values("validation_rmsle")
    plt.figure(figsize=(11, 8))
    plt.barh(plot_frame["model"], plot_frame["validation_rmsle"])
    plt.xlabel("Validation RMSLE")
    plt.ylabel("Approach")
    plt.title("Four-Week Forecasting Validation RMSLE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_rmsle_comparison.png", dpi=300)
    plt.close()

    actual = selected_predictions[target].to_numpy(float)
    predicted = selected_predictions["predicted_future_4week_stars"].to_numpy(float)
    upper = float(max(actual.max(), predicted.max()))
    plt.figure(figsize=(8, 7))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([0, upper], [0, upper], linestyle="--")
    plt.xlabel("Observed Future Four-Week Stars")
    plt.ylabel("Predicted Future Four-Week Stars")
    plt.title(f"Observed vs Predicted: {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_model_observed_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 7))
    plt.scatter(predicted, predicted - actual, alpha=0.8)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted Future Four-Week Stars")
    plt.ylabel("Forecast Error (Predicted - Observed)")
    plt.title(f"Validation Residuals: {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_model_validation_residuals.png", dpi=300)
    plt.close()

    p = importance.sort_values("importance_mean_rmsle_increase")
    plt.figure(figsize=(9, 7))
    plt.barh(p["feature"], p["importance_mean_rmsle_increase"], xerr=p["importance_std"])
    plt.xlabel("Increase in RMSLE After Permutation")
    plt.ylabel("Feature")
    plt.title("Selected Forecast Model Permutation Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_model_permutation_importance.png", dpi=300)
    plt.close()


def main():
    protected = [
        "all_hyperparameter_results.csv",
        "cv_fold_metrics.csv",
        "model_comparison.csv",
        "selected_model_metadata.json",
    ]
    existing = [name for name in protected if (OUTPUT_DIR / name).exists()]
    if existing:
        raise RuntimeError("Model-search outputs already exist; refusing to overwrite: " + ", ".join(existing))

    config = load_json(CONFIG_PATH)["forecasting"]
    schema = load_json(SCHEMA_PATH)
    features = schema["feature_columns"]
    target = schema["target"]

    train = validate_frame(pd.read_csv(TRAIN_PATH), "train", features, target)
    validation = validate_frame(pd.read_csv(VALIDATION_PATH), "validation", features, target)

    prepared, folds, fold_summary, _, eligibility = build_repository_aware_forecasting_cv(
        frame=train,
        target_column=target,
        n_splits=int(config["cross_validation_folds"]),
        validation_samples_per_repository=int(config["cross_validation_validation_samples_per_repository"]),
        gap_samples=int(config["cross_validation_purge_weeks"]),
        require_all_repositories=bool(config["cross_validation_require_all_repositories"]),
    )

    X = prepared[features]
    y = prepared[target]
    spaces = search_spaces()
    total = sum(len(spec["grid"]) * len(spec["transforms"]) for spec in spaces.values())

    print("=" * 110)
    print("FOUR-WEEK FORECASTING MODEL SEARCH")
    print("=" * 110)
    print(f"Training rows: {len(prepared)}")
    print(f"Training repositories: {prepared['repo_id'].nunique()}")
    print(f"External validation rows: {len(validation)}")
    print(f"External validation repositories: {validation['repo_id'].nunique()}")
    print(f"Features: {len(features)}")
    print(f"Temporal CV folds: {len(folds)}")
    print(f"Candidate configurations: {total}")
    print("Test data: not loaded")

    result_rows = []
    fold_rows = []
    configuration_id = 0

    for model_name, spec in spaces.items():
        print(f"\nSearching {model_name}...")
        success = 0
        failed = 0
        for transform in spec["transforms"]:
            for params in spec["grid"]:
                configuration_id += 1
                started = time.perf_counter()
                current_folds = []
                try:
                    for fold_number, (train_idx, val_idx) in enumerate(folds, 1):
                        estimator = build_estimator(model_name, params, transform)
                        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
                        predictions = np.clip(estimator.predict(X.iloc[val_idx]), 0.0, None)
                        metrics = evaluate(y.iloc[val_idx].to_numpy(float), predictions)
                        row = {
                            "configuration_id": configuration_id,
                            "model": model_name,
                            "target_transform": transform,
                            "parameters": json.dumps(params, sort_keys=True, default=str),
                            "fold": fold_number,
                            "train_rows": len(train_idx),
                            "validation_rows": len(val_idx),
                            **metrics,
                        }
                        current_folds.append(row)
                        fold_rows.append(row)

                    frame = pd.DataFrame(current_folds)
                    result_rows.append({
                        "configuration_id": configuration_id,
                        "model": model_name,
                        "target_transform": transform,
                        "parameters": json.dumps(params, sort_keys=True, default=str),
                        "successful_folds": len(frame),
                        "mean_rmsle": float(frame["rmsle"].mean()),
                        "std_rmsle": float(frame["rmsle"].std(ddof=1)),
                        "mean_mae": float(frame["mae"].mean()),
                        "std_mae": float(frame["mae"].std(ddof=1)),
                        "mean_rmse": float(frame["rmse"].mean()),
                        "mean_smape": float(frame["smape"].mean()),
                        "mean_forecast_bias": float(frame["forecast_bias"].mean()),
                        "elapsed_seconds": time.perf_counter() - started,
                        "status": "success",
                        "error": "",
                    })
                    success += 1
                except Exception as exc:
                    result_rows.append({
                        "configuration_id": configuration_id,
                        "model": model_name,
                        "target_transform": transform,
                        "parameters": json.dumps(params, sort_keys=True, default=str),
                        "successful_folds": 0,
                        "mean_rmsle": np.nan,
                        "std_rmsle": np.nan,
                        "mean_mae": np.nan,
                        "std_mae": np.nan,
                        "mean_rmse": np.nan,
                        "mean_smape": np.nan,
                        "mean_forecast_bias": np.nan,
                        "elapsed_seconds": time.perf_counter() - started,
                        "status": "failed",
                        "error": repr(exc),
                    })
                    failed += 1
        print(f"Completed {model_name}: {success} successful, {failed} failed.")

    results = pd.DataFrame(result_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    # CV baselines
    baseline_cv_rows = []
    for fold_number, (_, val_idx) in enumerate(folds, 1):
        fold_frame = prepared.iloc[val_idx]
        y_true = fold_frame[target].to_numpy(float)
        for name, predictions in baseline_predictions(fold_frame).items():
            baseline_cv_rows.append({
                "model": f"baseline_{name}",
                "fold": fold_number,
                "validation_rows": len(val_idx),
                **evaluate(y_true, predictions),
            })
    baseline_cv = pd.DataFrame(baseline_cv_rows)

    results.to_csv(OUTPUT_DIR / "all_hyperparameter_results.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)
    baseline_cv.to_csv(OUTPUT_DIR / "baseline_cv_fold_metrics.csv", index=False)
    results[results["status"] == "failed"].to_csv(OUTPUT_DIR / "failed_configurations.csv", index=False)

    successful = results[results["status"] == "success"].copy()
    if successful.empty:
        raise RuntimeError("Every trained forecasting configuration failed.")

    best_family = (
        successful.sort_values(["model", "mean_rmsle", "std_rmsle", "mean_mae"])
        .groupby("model", as_index=False)
        .first()
    )

    comparison_rows = []
    prediction_table = validation[["repo_id", "repo_full_name", "cutoff_week", "target_end_week", target]].copy()
    fitted = {}
    X_train = prepared[features]
    y_train = prepared[target]
    X_val = validation[features]
    y_val = validation[target].to_numpy(float)

    for row in best_family.itertuples(index=False):
        params = json.loads(row.parameters)
        estimator = build_estimator(row.model, params, row.target_transform)
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predictions = np.clip(estimator.predict(X_val), 0.0, None)
        metrics = evaluate(y_val, predictions)
        path = CANDIDATE_DIR / f"{row.model}.joblib"
        joblib.dump(estimator, path)
        fitted[row.model] = (estimator, params, row.target_transform, path)
        prediction_table[f"prediction_{row.model}"] = predictions
        comparison_rows.append({
            "approach_type": "trained_model",
            "model": row.model,
            "target_transform": row.target_transform,
            "parameters": row.parameters,
            "cv_mean_rmsle": float(row.mean_rmsle),
            "cv_std_rmsle": float(row.std_rmsle),
            "cv_mean_mae": float(row.mean_mae),
            "validation_rmsle": metrics["rmsle"],
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_smape": metrics["smape"],
            "validation_forecast_bias": metrics["forecast_bias"],
            "fit_seconds": fit_seconds,
            "artifact_path": str(path.relative_to(ROOT)),
        })

    for name, predictions in baseline_predictions(validation).items():
        metrics = evaluate(y_val, predictions)
        prediction_table[f"prediction_baseline_{name}"] = predictions
        comparison_rows.append({
            "approach_type": "naive_baseline",
            "model": f"baseline_{name}",
            "target_transform": "not_applicable",
            "parameters": "{}",
            "cv_mean_rmsle": np.nan,
            "cv_std_rmsle": np.nan,
            "cv_mean_mae": np.nan,
            "validation_rmsle": metrics["rmsle"],
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_smape": metrics["smape"],
            "validation_forecast_bias": metrics["forecast_bias"],
            "fit_seconds": 0.0,
            "artifact_path": "",
        })

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_rmsle", "validation_mae", "cv_mean_rmsle"], na_position="last"
    ).reset_index(drop=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    prediction_table.to_csv(OUTPUT_DIR / "validation_predictions_all_models.csv", index=False)

    trained = comparison[comparison["approach_type"] == "trained_model"].sort_values(
        ["validation_rmsle", "validation_mae", "cv_mean_rmsle", "cv_std_rmsle"]
    )
    selected_row = trained.iloc[0]
    selected_name = selected_row["model"]
    selected_estimator, selected_params, selected_transform, _ = fitted[selected_name]
    selected_predictions_array = np.clip(selected_estimator.predict(X_val), 0.0, None)

    selected_predictions = validation[["repo_id", "repo_full_name", "cutoff_week", "target_end_week", target]].copy()
    selected_predictions["predicted_future_4week_stars"] = selected_predictions_array
    selected_predictions["forecast_error"] = selected_predictions_array - y_val
    selected_predictions["absolute_error"] = np.abs(selected_predictions["forecast_error"])
    selected_predictions.to_csv(OUTPUT_DIR / "selected_model_validation_predictions.csv", index=False)

    def scorer(estimator, X_score, y_score):
        return -rmsle(y_score, np.clip(estimator.predict(X_score), 0.0, None))

    perm = permutation_importance(
        selected_estimator,
        X_val,
        y_val,
        scoring=scorer,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": features,
        "importance_mean_rmsle_increase": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean_rmsle_increase", ascending=False)
    importance.to_csv(OUTPUT_DIR / "selected_model_permutation_importance.csv", index=False)

    selected_model_path = MODEL_DIR / "selected_forecaster_development.joblib"
    joblib.dump(selected_estimator, selected_model_path)

    best_baseline = comparison[comparison["approach_type"] == "naive_baseline"].sort_values("validation_rmsle").iloc[0]
    selected_metrics = {
        "rmsle": float(selected_row["validation_rmsle"]),
        "mae": float(selected_row["validation_mae"]),
        "rmse": float(selected_row["validation_rmse"]),
        "smape": float(selected_row["validation_smape"]),
        "forecast_bias": float(selected_row["validation_forecast_bias"]),
    }
    baseline_metrics = {
        "name": best_baseline["model"],
        "rmsle": float(best_baseline["validation_rmsle"]),
        "mae": float(best_baseline["validation_mae"]),
        "rmse": float(best_baseline["validation_rmse"]),
        "smape": float(best_baseline["validation_smape"]),
        "forecast_bias": float(best_baseline["validation_forecast_bias"]),
    }

    metadata = {
        "development_status": "forecasting_model_search_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": schema["task"],
        "target": target,
        "features": features,
        "feature_count": len(features),
        "training_rows": len(prepared),
        "training_repositories": int(prepared["repo_id"].nunique()),
        "external_validation_rows": len(validation),
        "external_validation_repositories": int(validation["repo_id"].nunique()),
        "temporal_cv": {
            "folds": len(folds),
            "purge_weeks": int(config["cross_validation_purge_weeks"]),
            "validation_samples_per_repository": int(config["cross_validation_validation_samples_per_repository"]),
            "all_repositories_eligible": bool(eligibility["eligible"].all()),
            "fold_summary": fold_summary.to_dict(orient="records"),
        },
        "candidate_configurations": total,
        "successful_configurations": int((results["status"] == "success").sum()),
        "failed_configurations": int((results["status"] == "failed").sum()),
        "selected_model": selected_name,
        "selected_target_transform": selected_transform,
        "selected_parameters": selected_params,
        "selected_cv_mean_rmsle": float(selected_row["cv_mean_rmsle"]),
        "selected_cv_std_rmsle": float(selected_row["cv_std_rmsle"]),
        "selected_validation_metrics": selected_metrics,
        "best_validation_baseline": baseline_metrics,
        "improvement_over_best_validation_baseline": {
            "absolute_rmsle_reduction": baseline_metrics["rmsle"] - selected_metrics["rmsle"],
            "relative_rmsle_reduction": (baseline_metrics["rmsle"] - selected_metrics["rmsle"]) / baseline_metrics["rmsle"],
            "absolute_mae_reduction": baseline_metrics["mae"] - selected_metrics["mae"],
        },
        "selection_rule": (
            "Choose the best CV-tuned trained model family by external validation RMSLE; "
            "break ties using validation MAE, CV mean RMSLE, and CV RMSLE standard deviation."
        ),
        "selected_model_path": str(selected_model_path.relative_to(ROOT)),
        "test_data_status": "not_loaded_or_evaluated",
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    (OUTPUT_DIR / "selected_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (MODEL_DIR / "selected_forecaster_development_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    save_plots(comparison, selected_predictions, importance, target, selected_name)

    report = "\n".join([
        "=" * 110,
        "FOUR-WEEK FORECASTING MODEL-SEARCH REPORT",
        "=" * 110,
        "",
        f"Training rows: {len(prepared)}",
        f"Training repositories: {prepared['repo_id'].nunique()}",
        f"External validation rows: {len(validation)}",
        f"External validation repositories: {validation['repo_id'].nunique()}",
        f"Candidate configurations: {total}",
        f"Successful configurations: {(results['status'] == 'success').sum()}",
        f"Failed configurations: {(results['status'] == 'failed').sum()}",
        "",
        f"Selected model: {selected_name}",
        f"Selected target transform: {selected_transform}",
        f"Selected parameters: {json.dumps(selected_params, sort_keys=True, default=str)}",
        f"Selected CV mean RMSLE: {float(selected_row['cv_mean_rmsle']):.6f}",
        f"Selected CV RMSLE SD: {float(selected_row['cv_std_rmsle']):.6f}",
        "",
        "SELECTED MODEL VALIDATION METRICS",
        "-" * 110,
        json.dumps(selected_metrics, indent=2),
        "",
        "BEST NAIVE VALIDATION BASELINE",
        "-" * 110,
        json.dumps(baseline_metrics, indent=2),
        "",
        "MODEL COMPARISON",
        "-" * 110,
        comparison.to_string(index=False),
        "",
        "PERMUTATION IMPORTANCE",
        "-" * 110,
        importance.to_string(index=False),
        "",
        "Forecast test data was not loaded or evaluated.",
    ])
    (OUTPUT_DIR / "forecasting_model_search_report.txt").write_text(report, encoding="utf-8")

    print("\n" + "=" * 110)
    print("MODEL COMPARISON")
    print("=" * 110)
    print(comparison.to_string(index=False))
    print("\n" + "=" * 110)
    print("SELECTED MODEL")
    print("=" * 110)
    print(json.dumps(metadata, indent=2, default=str))
    print("\n" + "=" * 110)
    print("PERMUTATION IMPORTANCE")
    print("=" * 110)
    print(importance.to_string(index=False))
    print("\nForecast test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
