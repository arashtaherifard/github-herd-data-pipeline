import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_config.json"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_classification_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return config["classification"]


def build_repository_aware_temporal_cv(
    frame: pd.DataFrame,
    target_column: str,
    n_splits: int,
    validation_samples_per_repository: int,
    gap_samples: int,
    require_all_repositories: bool = True,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
]:
    required_columns = {
        "repo_id",
        "repo_full_name",
        "cutoff_week",
        "target_end_week",
        target_column,
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classification data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = frame.copy()

    prepared["cutoff_week"] = pd.to_datetime(
        prepared["cutoff_week"],
        utc=True,
        errors="raise",
    )

    prepared["target_end_week"] = pd.to_datetime(
        prepared["target_end_week"],
        utc=True,
        errors="raise",
    )

    prepared = prepared.sort_values(
        ["repo_id", "cutoff_week"]
    ).reset_index(drop=True)

    fold_storage = {
        fold_number: {
            "train_positions": [],
            "validation_positions": [],
            "repositories": set(),
        }
        for fold_number in range(
            1,
            n_splits + 1,
        )
    }

    repository_rows = []

    for repo_id, group in prepared.groupby(
        "repo_id",
        sort=True,
    ):
        positions = group.index.to_numpy()

        splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=(
                validation_samples_per_repository
            ),
            gap=gap_samples,
        )

        try:
            repository_splits = list(
                splitter.split(positions)
            )
            eligible = True
            error = ""
        except ValueError as exception:
            repository_splits = []
            eligible = False
            error = str(exception)

        repository_rows.append(
            {
                "repo_id": repo_id,
                "repo_full_name": (
                    group["repo_full_name"].iloc[0]
                ),
                "rows": int(len(group)),
                "eligible": eligible,
                "error": error,
            }
        )

        if not eligible:
            continue

        for fold_number, (
            local_train,
            local_validation,
        ) in enumerate(
            repository_splits,
            start=1,
        ):
            fold_storage[fold_number][
                "train_positions"
            ].extend(
                positions[local_train].tolist()
            )

            fold_storage[fold_number][
                "validation_positions"
            ].extend(
                positions[
                    local_validation
                ].tolist()
            )

            fold_storage[fold_number][
                "repositories"
            ].add(repo_id)

    repository_eligibility = pd.DataFrame(
        repository_rows
    )

    ineligible = repository_eligibility[
        ~repository_eligibility["eligible"]
    ]

    if (
        require_all_repositories
        and not ineligible.empty
    ):
        raise ValueError(
            "The selected temporal CV configuration "
            "does not support all repositories:\n"
            + ineligible.to_string(index=False)
        )

    folds = []
    summary_rows = []
    assignment_rows = []

    for fold_number, stored in (
        fold_storage.items()
    ):
        train_positions = np.asarray(
            sorted(stored["train_positions"]),
            dtype=int,
        )

        validation_positions = np.asarray(
            sorted(
                stored[
                    "validation_positions"
                ]
            ),
            dtype=int,
        )

        if len(train_positions) == 0:
            raise ValueError(
                f"Fold {fold_number} has no "
                "training observations."
            )

        if len(validation_positions) == 0:
            raise ValueError(
                f"Fold {fold_number} has no "
                "validation observations."
            )

        train = prepared.iloc[
            train_positions
        ]

        validation = prepared.iloc[
            validation_positions
        ]

        if train[target_column].nunique() != 2:
            raise ValueError(
                f"Fold {fold_number} training "
                "data does not contain both classes."
            )

        if (
            validation[
                target_column
            ].nunique()
            != 2
        ):
            raise ValueError(
                f"Fold {fold_number} validation "
                "data does not contain both classes."
            )

        overlapping_positions = set(
            train_positions
        ) & set(validation_positions)

        if overlapping_positions:
            raise ValueError(
                f"Fold {fold_number} has direct "
                "row overlap."
            )

        target_window_overlap_violations = 0

        shared_repositories = (
            set(train["repo_id"])
            & set(validation["repo_id"])
        )

        for repo_id in shared_repositories:
            repository_train = train[
                train["repo_id"] == repo_id
            ]

            repository_validation = validation[
                validation["repo_id"] == repo_id
            ]

            latest_training_target_end = (
                repository_train[
                    "target_end_week"
                ].max()
            )

            earliest_validation_cutoff = (
                repository_validation[
                    "cutoff_week"
                ].min()
            )

            if (
                latest_training_target_end
                >= earliest_validation_cutoff
            ):
                target_window_overlap_violations += 1

        if target_window_overlap_violations:
            raise ValueError(
                f"Fold {fold_number} contains "
                f"{target_window_overlap_violations} "
                "future-target-window overlap "
                "violations."
            )

        folds.append(
            (
                train_positions,
                validation_positions,
            )
        )

        summary_rows.append(
            {
                "fold": fold_number,
                "repositories": int(
                    len(stored["repositories"])
                ),
                "train_rows": int(
                    len(train_positions)
                ),
                "validation_rows": int(
                    len(validation_positions)
                ),
                "train_negative_rows": int(
                    (
                        train[target_column]
                        == 0
                    ).sum()
                ),
                "train_positive_rows": int(
                    (
                        train[target_column]
                        == 1
                    ).sum()
                ),
                "validation_negative_rows": int(
                    (
                        validation[
                            target_column
                        ]
                        == 0
                    ).sum()
                ),
                "validation_positive_rows": int(
                    (
                        validation[
                            target_column
                        ]
                        == 1
                    ).sum()
                ),
                "train_positive_rate": float(
                    train[target_column].mean()
                ),
                "validation_positive_rate": float(
                    validation[
                        target_column
                    ].mean()
                ),
                "target_window_overlap_violations": (
                    target_window_overlap_violations
                ),
            }
        )

        for role, positions_for_role in [
            ("train", train_positions),
            (
                "validation",
                validation_positions,
            ),
        ]:
            selected_rows = prepared.iloc[
                positions_for_role
            ]

            for position, row in zip(
                positions_for_role,
                selected_rows.itertuples(
                    index=False
                ),
            ):
                assignment_rows.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "row_position": int(
                            position
                        ),
                        "repo_id": row.repo_id,
                        "repo_full_name": (
                            row.repo_full_name
                        ),
                        "cutoff_week": str(
                            row.cutoff_week
                        ),
                        "target_end_week": str(
                            row.target_end_week
                        ),
                        "target": int(
                            getattr(
                                row,
                                target_column,
                            )
                        ),
                    }
                )

    fold_summary = pd.DataFrame(
        summary_rows
    )

    assignments = pd.DataFrame(
        assignment_rows
    )

    return (
        prepared,
        folds,
        fold_summary,
        assignments,
    )


def main() -> None:
    settings = load_classification_config()

    target = settings["target"]

    frame = pd.read_csv(TRAIN_PATH)

    (
        prepared,
        folds,
        fold_summary,
        assignments,
    ) = build_repository_aware_temporal_cv(
        frame=frame,
        target_column=target,
        n_splits=int(
            settings[
                "cross_validation_folds"
            ]
        ),
        validation_samples_per_repository=int(
            settings[
                "cross_validation_"
                "validation_samples_per_repository"
            ]
        ),
        gap_samples=int(
            settings["purge_weeks"]
        ),
        require_all_repositories=bool(
            settings[
                "cross_validation_"
                "require_all_repositories"
            ]
        ),
    )

    fold_summary.to_csv(
        OUTPUT_DIR
        / "selected_temporal_cv_summary.csv",
        index=False,
    )

    assignments.to_csv(
        OUTPUT_DIR
        / "selected_temporal_cv_assignments.csv",
        index=False,
    )

    metadata = {
        "target": target,
        "training_rows": int(
            len(prepared)
        ),
        "repositories": int(
            prepared["repo_id"].nunique()
        ),
        "folds": len(folds),
        "validation_samples_per_repository": int(
            settings[
                "cross_validation_"
                "validation_samples_per_repository"
            ]
        ),
        "purge_gap_weeks": int(
            settings["purge_weeks"]
        ),
        "all_repositories_required": bool(
            settings[
                "cross_validation_"
                "require_all_repositories"
            ]
        ),
        "test_status": (
            "not_loaded_or_evaluated"
        ),
    }

    (
        OUTPUT_DIR
        / "selected_temporal_cv_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SELECTED CLASSIFICATION TEMPORAL CV")
    print("=" * 100)
    print(json.dumps(metadata, indent=2))

    print("\n" + "=" * 100)
    print("FOLD SUMMARY")
    print("=" * 100)
    print(
        fold_summary.to_string(index=False)
    )

    print(
        "\nTest data was not loaded "
        "or evaluated."
    )


if __name__ == "__main__":
    main()
