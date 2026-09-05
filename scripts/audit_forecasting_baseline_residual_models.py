import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forecasting_temporal_cv import (
    build_repository_aware_forecasting_cv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_config.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecasting_schema.json"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "baseline_residual_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        a_min=0.0,
        a_max=None,
    )

    return float(
        np.sqrt(
            mean_squared_log_error(
                y_true,
                y_pred,
            )
        )
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    numerator = (
        2.0
        * np.abs(
            y_pred - y_true
        )
    )

    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )

    return float(
        np.mean(values)
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        a_min=0.0,
        a_max=None,
    )

    return {
        "rmsle": rmsle(
            y_true,
            y_pred,
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": rmse(
            y_true,
            y_pred,
        ),
        "smape": smape(
            y_true,
            y_pred,
        ),
        "forecast_bias": float(
            np.mean(
                y_pred - y_true
            )
        ),
    }


def baseline_prediction(
    frame: pd.DataFrame,
) -> np.ndarray:
    return (
        4.0
        * frame[
            "rolling_3week_mean_stars"
        ].to_numpy(dtype=float)
    )


def validate_frame(
    frame: pd.DataFrame,
    split_name: str,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    required = set(
        features
        + [
            "repo_id",
            "repo_full_name",
            "cutoff_week",
            "target_end_week",
            target,
        ]
    )

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{split_name} is missing: "
            f"{sorted(missing)}"
        )

    prepared = frame.copy()

    prepared["cutoff_week"] = (
        pd.to_datetime(
            prepared["cutoff_week"],
            utc=True,
            errors="raise",
        )
    )

    prepared["target_end_week"] = (
        pd.to_datetime(
            prepared["target_end_week"],
            utc=True,
            errors="raise",
        )
    )

    prepared[features] = prepared[
        features
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    prepared[target] = pd.to_numeric(
        prepared[target],
        errors="raise",
    )

    if prepared[features].isna().any().any():
        raise ValueError(
            f"{split_name} contains missing "
            "feature values."
        )

    if not np.isfinite(
        prepared[
            features
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{split_name} contains non-finite "
            "feature values."
        )

    if (
        prepared[target].isna().any()
        or (prepared[target] < 0).any()
    ):
        raise ValueError(
            f"{split_name} has an invalid target."
        )

    return prepared


def build_model(
    model_kind: str,
    params: dict[str, Any],
) -> Any:
    if model_kind == "ridge":
        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "regressor",
                    Ridge(
                        alpha=float(
                            params["alpha"]
                        )
                    ),
                ),
            ]
        )

    if model_kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=int(
                params["max_depth"]
            ),
            min_samples_leaf=int(
                params["min_samples_leaf"]
            ),
            max_features=params[
                "max_features"
            ],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown model kind: {model_kind}"
    )


def model_prediction(
    mode: str,
    estimator: Any,
    train_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    features: list[str],
    target: str,
    blend_weight: float | None = None,
) -> np.ndarray:
    train_baseline = baseline_prediction(
        train_frame
    )

    prediction_baseline = (
        baseline_prediction(
            prediction_frame
        )
    )

    train_y = train_frame[
        target
    ].to_numpy(dtype=float)

    if mode == "direct":
        estimator.fit(
            train_frame[features],
            train_y,
        )

        prediction = estimator.predict(
            prediction_frame[features]
        )

    elif mode == "additive_residual":
        residual_target = (
            train_y - train_baseline
        )

        estimator.fit(
            train_frame[features],
            residual_target,
        )

        prediction = (
            prediction_baseline
            + estimator.predict(
                prediction_frame[features]
            )
        )

    elif mode == "log_residual":
        residual_target = (
            np.log1p(train_y)
            - np.log1p(
                train_baseline
            )
        )

        estimator.fit(
            train_frame[features],
            residual_target,
        )

        correction = estimator.predict(
            prediction_frame[features]
        )

        prediction = np.expm1(
            np.log1p(
                prediction_baseline
            )
            + correction
        )

    elif mode == "blend_direct":
        if blend_weight is None:
            raise ValueError(
                "blend_weight is required."
            )

        estimator.fit(
            train_frame[features],
            train_y,
        )

        direct_prediction = (
            estimator.predict(
                prediction_frame[features]
            )
        )

        prediction = (
            blend_weight
            * direct_prediction
            + (
                1.0
                - blend_weight
            )
            * prediction_baseline
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )


def candidate_definitions(
    feature_sets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    candidates = []

    for scale in [
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    ]:
        candidates.append(
            {
                "family": (
                    "scaled_rolling_baseline"
                ),
                "mode": (
                    "scaled_baseline"
                ),
                "model_kind": (
                    "none"
                ),
                "feature_set": (
                    "baseline_only"
                ),
                "parameters": {
                    "scale": scale,
                },
            }
        )

    for feature_set_name in [
        "full",
        "recent_only",
        "recent_compact",
    ]:
        for alpha in [
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
        ]:
            for mode in [
                "additive_residual",
                "log_residual",
            ]:
                candidates.append(
                    {
                        "family": (
                            f"{mode}_ridge"
                        ),
                        "mode": mode,
                        "model_kind": (
                            "ridge"
                        ),
                        "feature_set": (
                            feature_set_name
                        ),
                        "parameters": {
                            "alpha": alpha,
                        },
                    }
                )

    extra_tree_grids = [
        {
            "max_depth": 4,
            "min_samples_leaf": 5,
            "max_features": 1.0,
        },
        {
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": 1.0,
        },
        {
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": 1.0,
        },
        {
            "max_depth": 4,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
    ]

    for feature_set_name in [
        "full",
        "recent_only",
        "recent_compact",
    ]:
        for parameters in (
            extra_tree_grids
        ):
            for mode in [
                "direct",
                "additive_residual",
                "log_residual",
            ]:
                candidates.append(
                    {
                        "family": (
                            f"{mode}_extra_trees"
                        ),
                        "mode": mode,
                        "model_kind": (
                            "extra_trees"
                        ),
                        "feature_set": (
                            feature_set_name
                        ),
                        "parameters": (
                            parameters.copy()
                        ),
                    }
                )

            for weight in [
                0.2,
                0.4,
                0.6,
                0.8,
            ]:
                parameters_with_weight = (
                    parameters.copy()
                )

                parameters_with_weight[
                    "blend_weight"
                ] = weight

                candidates.append(
                    {
                        "family": (
                            "blend_direct_"
                            "extra_trees"
                        ),
                        "mode": (
                            "blend_direct"
                        ),
                        "model_kind": (
                            "extra_trees"
                        ),
                        "feature_set": (
                            feature_set_name
                        ),
                        "parameters": (
                            parameters_with_weight
                        ),
                    }
                )

    return candidates


def evaluate_candidate_cv(
    candidate_id: int,
    candidate: dict[str, Any],
    prepared_train: pd.DataFrame,
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    feature_sets: dict[str, list[str]],
    target: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    fold_rows = []

    start = time.perf_counter()

    for fold_number, (
        train_positions,
        validation_positions,
    ) in enumerate(
        folds,
        start=1,
    ):
        fold_train = (
            prepared_train.iloc[
                train_positions
            ]
        )

        fold_validation = (
            prepared_train.iloc[
                validation_positions
            ]
        )

        y_true = fold_validation[
            target
        ].to_numpy(dtype=float)

        if (
            candidate["mode"]
            == "scaled_baseline"
        ):
            predictions = (
                float(
                    candidate[
                        "parameters"
                    ]["scale"]
                )
                * baseline_prediction(
                    fold_validation
                )
            )

        else:
            parameters = (
                candidate[
                    "parameters"
                ].copy()
            )

            blend_weight = (
                parameters.pop(
                    "blend_weight",
                    None,
                )
            )

            estimator = build_model(
                candidate[
                    "model_kind"
                ],
                parameters,
            )

            features = feature_sets[
                candidate[
                    "feature_set"
                ]
            ]

            predictions = (
                model_prediction(
                    mode=candidate[
                        "mode"
                    ],
                    estimator=estimator,
                    train_frame=(
                        fold_train
                    ),
                    prediction_frame=(
                        fold_validation
                    ),
                    features=features,
                    target=target,
                    blend_weight=(
                        blend_weight
                    ),
                )
            )

        metrics = evaluate_predictions(
            y_true,
            predictions,
        )

        fold_rows.append(
            {
                "candidate_id": (
                    candidate_id
                ),
                "family": (
                    candidate["family"]
                ),
                "mode": (
                    candidate["mode"]
                ),
                "model_kind": (
                    candidate[
                        "model_kind"
                    ]
                ),
                "feature_set": (
                    candidate[
                        "feature_set"
                    ]
                ),
                "parameters": json.dumps(
                    candidate[
                        "parameters"
                    ],
                    sort_keys=True,
                ),
                "fold": fold_number,
                **metrics,
            }
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    fold_frame = pd.DataFrame(
        fold_rows
    )

    summary = {
        "candidate_id": (
            candidate_id
        ),
        "family": (
            candidate["family"]
        ),
        "mode": (
            candidate["mode"]
        ),
        "model_kind": (
            candidate["model_kind"]
        ),
        "feature_set": (
            candidate["feature_set"]
        ),
        "parameters": json.dumps(
            candidate["parameters"],
            sort_keys=True,
        ),
        "mean_cv_rmsle": float(
            fold_frame[
                "rmsle"
            ].mean()
        ),
        "std_cv_rmsle": float(
            fold_frame[
                "rmsle"
            ].std(ddof=1)
        ),
        "mean_cv_mae": float(
            fold_frame[
                "mae"
            ].mean()
        ),
        "mean_cv_rmse": float(
            fold_frame[
                "rmse"
            ].mean()
        ),
        "mean_cv_smape": float(
            fold_frame[
                "smape"
            ].mean()
        ),
        "mean_cv_bias": float(
            fold_frame[
                "forecast_bias"
            ].mean()
        ),
        "elapsed_seconds": float(
            elapsed
        ),
    }

    return summary, fold_rows


def evaluate_family_winner(
    winner: pd.Series,
    prepared_train: pd.DataFrame,
    external_validation: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
) -> tuple[
    dict[str, Any],
    np.ndarray,
]:
    parameters = json.loads(
        winner["parameters"]
    )

    if (
        winner["mode"]
        == "scaled_baseline"
    ):
        predictions = (
            float(
                parameters["scale"]
            )
            * baseline_prediction(
                external_validation
            )
        )

    else:
        blend_weight = (
            parameters.pop(
                "blend_weight",
                None,
            )
        )

        estimator = build_model(
            winner["model_kind"],
            parameters,
        )

        predictions = model_prediction(
            mode=winner["mode"],
            estimator=estimator,
            train_frame=(
                prepared_train
            ),
            prediction_frame=(
                external_validation
            ),
            features=feature_sets[
                winner["feature_set"]
            ],
            target=target,
            blend_weight=(
                blend_weight
            ),
        )

    metrics = evaluate_predictions(
        external_validation[
            target
        ].to_numpy(dtype=float),
        predictions,
    )

    result = {
        "family": winner["family"],
        "mode": winner["mode"],
        "model_kind": (
            winner["model_kind"]
        ),
        "feature_set": (
            winner["feature_set"]
        ),
        "parameters": (
            winner["parameters"]
        ),
        "cv_mean_rmsle": float(
            winner[
                "mean_cv_rmsle"
            ]
        ),
        "cv_std_rmsle": float(
            winner[
                "std_cv_rmsle"
            ]
        ),
        "cv_mean_mae": float(
            winner[
                "mean_cv_mae"
            ]
        ),
        "validation_rmsle": (
            metrics["rmsle"]
        ),
        "validation_mae": (
            metrics["mae"]
        ),
        "validation_rmse": (
            metrics["rmse"]
        ),
        "validation_smape": (
            metrics["smape"]
        ),
        "validation_forecast_bias": (
            metrics[
                "forecast_bias"
            ]
        ),
    }

    return result, predictions


def main() -> None:
    protected = (
        OUTPUT_DIR
        / "validation_family_comparison.csv"
    )

    if protected.exists():
        raise RuntimeError(
            "Audit outputs already exist. "
            "Refusing to overwrite."
        )

    config = load_json(
        CONFIG_PATH
    )["forecasting"]

    schema = load_json(
        SCHEMA_PATH
    )

    target = schema["target"]

    full_features = schema[
        "feature_columns"
    ]

    feature_sets = {
        "full": full_features,
        "recent_only": [
            "weekly_new_stars",
            "previous_week_stars",
            "weekly_growth_rate",
            "lag_2_week_stars",
            "rolling_3week_mean_stars",
            "rolling_4week_sum_stars",
            "weekly_growth_acceleration",
        ],
        "recent_compact": [
            "weekly_new_stars",
            "previous_week_stars",
            "lag_2_week_stars",
            "rolling_3week_mean_stars",
            "weekly_growth_acceleration",
        ],
    }

    all_features = sorted(
        {
            feature
            for values
            in feature_sets.values()
            for feature in values
        }
    )

    train = validate_frame(
        pd.read_csv(TRAIN_PATH),
        "train",
        all_features,
        target,
    )

    validation = validate_frame(
        pd.read_csv(
            VALIDATION_PATH
        ),
        "validation",
        all_features,
        target,
    )

    (
        prepared_train,
        folds,
        fold_summary,
        _,
        eligibility,
    ) = build_repository_aware_forecasting_cv(
        frame=train,
        target_column=target,
        n_splits=int(
            config[
                "cross_validation_folds"
            ]
        ),
        validation_samples_per_repository=int(
            config[
                "cross_validation_"
                "validation_samples_per_repository"
            ]
        ),
        gap_samples=int(
            config[
                "cross_validation_purge_weeks"
            ]
        ),
        require_all_repositories=True,
    )

    candidates = candidate_definitions(
        feature_sets
    )

    print("=" * 108)
    print(
        "FORECAST BASELINE, ABLATION, "
        "AND RESIDUAL-MODEL AUDIT"
    )
    print("=" * 108)

    print(
        "Training rows:",
        len(prepared_train),
    )

    print(
        "External validation rows:",
        len(validation),
    )

    print(
        "Candidate configurations:",
        len(candidates),
    )

    print(
        "Test data:",
        "not loaded",
    )

    summaries = []
    fold_rows = []

    for candidate_id, candidate in enumerate(
        candidates,
        start=1,
    ):
        summary, rows = (
            evaluate_candidate_cv(
                candidate_id=(
                    candidate_id
                ),
                candidate=candidate,
                prepared_train=(
                    prepared_train
                ),
                folds=folds,
                feature_sets=(
                    feature_sets
                ),
                target=target,
            )
        )

        summaries.append(summary)
        fold_rows.extend(rows)

    candidate_summary = pd.DataFrame(
        summaries
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    family_winners = (
        candidate_summary.sort_values(
            [
                "family",
                "mean_cv_rmsle",
                "std_cv_rmsle",
                "mean_cv_mae",
            ]
        )
        .groupby(
            "family",
            as_index=False,
        )
        .first()
    )

    validation_rows = []
    validation_predictions = (
        validation[
            [
                "repo_id",
                "repo_full_name",
                "cutoff_week",
                "target_end_week",
                target,
            ]
        ].copy()
    )

    raw_baseline = (
        baseline_prediction(
            validation
        )
    )

    raw_metrics = evaluate_predictions(
        validation[
            target
        ].to_numpy(dtype=float),
        raw_baseline,
    )

    validation_rows.append(
        {
            "family": (
                "raw_rolling_baseline"
            ),
            "mode": (
                "raw_baseline"
            ),
            "model_kind": "none",
            "feature_set": (
                "baseline_only"
            ),
            "parameters": (
                '{"scale": 1.0}'
            ),
            "cv_mean_rmsle": float(
                config[
                    "cross_validation_baseline"
                ][
                    "mean_rmsle"
                ]
            ),
            "cv_std_rmsle": float(
                config[
                    "cross_validation_baseline"
                ][
                    "std_rmsle"
                ]
            ),
            "cv_mean_mae": float(
                config[
                    "cross_validation_baseline"
                ][
                    "mean_mae"
                ]
            ),
            "validation_rmsle": (
                raw_metrics["rmsle"]
            ),
            "validation_mae": (
                raw_metrics["mae"]
            ),
            "validation_rmse": (
                raw_metrics["rmse"]
            ),
            "validation_smape": (
                raw_metrics["smape"]
            ),
            "validation_forecast_bias": (
                raw_metrics[
                    "forecast_bias"
                ]
            ),
        }
    )

    validation_predictions[
        "prediction_raw_rolling_baseline"
    ] = raw_baseline

    for winner in (
        family_winners.itertuples(
            index=False
        )
    ):
        result, predictions = (
            evaluate_family_winner(
                winner=pd.Series(
                    winner._asdict()
                ),
                prepared_train=(
                    prepared_train
                ),
                external_validation=(
                    validation
                ),
                feature_sets=(
                    feature_sets
                ),
                target=target,
            )
        )

        validation_rows.append(
            result
        )

        safe_family = (
            result["family"]
            .replace(" ", "_")
        )

        validation_predictions[
            f"prediction_{safe_family}"
        ] = predictions

    validation_comparison = (
        pd.DataFrame(
            validation_rows
        )
        .sort_values(
            [
                "validation_rmsle",
                "validation_mae",
                "cv_mean_rmsle",
            ]
        )
        .reset_index(drop=True)
    )

    selected = (
        validation_comparison.iloc[0]
    )

    candidate_summary.to_csv(
        OUTPUT_DIR
        / "all_candidate_cv_results.csv",
        index=False,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "candidate_cv_fold_metrics.csv",
        index=False,
    )

    family_winners.to_csv(
        OUTPUT_DIR
        / "cv_family_winners.csv",
        index=False,
    )

    validation_comparison.to_csv(
        OUTPUT_DIR
        / "validation_family_comparison.csv",
        index=False,
    )

    validation_predictions.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    metadata = {
        "status": (
            "baseline_residual_audit_complete"
        ),
        "target": target,
        "training_rows": int(
            len(prepared_train)
        ),
        "external_validation_rows": int(
            len(validation)
        ),
        "repositories": int(
            prepared_train[
                "repo_id"
            ].nunique()
        ),
        "candidate_configurations": int(
            len(candidates)
        ),
        "families": sorted(
            candidate_summary[
                "family"
            ].unique().tolist()
        ),
        "feature_sets": (
            feature_sets
        ),
        "selected_development_approach": (
            selected.to_dict()
        ),
        "selection_rule": (
            "Tune each approach family only "
            "with the frozen purged temporal CV, "
            "then compare one CV-selected winner "
            "per family on the external validation "
            "set. The deterministic rolling "
            "baseline is eligible for final "
            "selection."
        ),
        "fold_summary": (
            fold_summary.to_dict(
                orient="records"
            )
        ),
        "all_repositories_eligible": bool(
            eligibility[
                "eligible"
            ].all()
        ),
        "test_data_status": (
            "not_loaded_or_evaluated"
        ),
    }

    (
        OUTPUT_DIR
        / "audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print(
        "CV-SELECTED FAMILY WINNERS"
    )
    print("=" * 108)

    print(
        family_winners.to_string(
            index=False
        )
    )

    print()
    print("=" * 108)
    print(
        "EXTERNAL VALIDATION COMPARISON"
    )
    print("=" * 108)

    print(
        validation_comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 108)
    print(
        "SELECTED DEVELOPMENT APPROACH"
    )
    print("=" * 108)

    print(
        json.dumps(
            selected.to_dict(),
            indent=2,
            default=str,
        )
    )

    print()
    print(
        "Forecast test data was not "
        "loaded or evaluated."
    )


if __name__ == "__main__":
    main()
