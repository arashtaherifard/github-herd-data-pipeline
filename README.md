# GitHub Herd Behavior, Popularity Forecasting, and Repository Recommendation

**Introduction to Data Science — Final Project, Phase 3**

**Team members:**
- Arash Taherifard — `810102474`
- Shayan Maleki — `810102515`
- Ali Khosravi — `810102606`

**Repository:** https://github.com/arashtaherifard/github-herd-data-pipeline<br>
**Phase 3 branch:** https://github.com/arashtaherifard/github-herd-data-pipeline/tree/phase3<br>
**Current validated branch:** `phase3`<br>
**Validated commit:** `8a864558bc8ea79543a88bd587948cda3a3df087`

> **Submission note:** Replace the placeholders in the [Presentation and report](#presentation-and-report) section with the final Google Drive and report links before submission.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Motivation and real-world value](#motivation-and-real-world-value)
3. [Project objectives](#project-objectives)
4. [Dataset](#dataset)
5. [Machine-learning tasks](#machine-learning-tasks)
6. [Repository structure](#repository-structure)
7. [Pipeline architecture](#pipeline-architecture)
8. [Environment setup](#environment-setup)
9. [How to run the project](#how-to-run-the-project)
10. [Model development and scientific protocol](#model-development-and-scientific-protocol)
11. [Final model results](#final-model-results)
12. [Operational predictions](#operational-predictions)
13. [Database integration](#database-integration)
14. [Prefect orchestration](#prefect-orchestration)
15. [MLflow model management](#mlflow-model-management)
16. [Docker, Kubernetes, and CI/CD](#docker-kubernetes-and-cicd)
17. [Reproducibility and safety](#reproducibility-and-safety)
18. [Real-world interpretation](#real-world-interpretation)
19. [Limitations](#limitations)
20. [Presentation and report](#presentation-and-report)
21. [Validated Git references](#validated-git-references)

---

## Project overview

This project studies **herd behavior and collective attention in the GitHub open-source ecosystem**. It combines repository metadata, historical star activity, user–repository interactions, topics, programming languages, temporal growth signals, and graph structure to build an end-to-end data science system.

The final system solves three related problems:

1. **Four-week popularity forecasting**<br>
   Predict the number of stars a repository is expected to receive over the next 28 days.

2. **High-growth classification**<br>
   Estimate whether a repository is likely to experience a future growth surge.

3. **Herd-aware repository recommendation**<br>
   Recommend unseen repositories to users while balancing relevance, catalog coverage, novelty, and the risk of reinforcing existing popularity.

The project includes the complete workflow from data collection and database construction to preprocessing, feature engineering, model selection, one-time final evaluation, operational prediction generation, database persistence, MLflow registration, Prefect orchestration, Docker packaging, Kubernetes execution, and GitHub Actions validation.

---

## Motivation and real-world value

GitHub stars are often used as a public signal of repository visibility and adoption. However, popularity is highly concentrated, and attention can become self-reinforcing: visible repositories attract more users, which increases their visibility further. This creates both useful signals and potential herd effects.

The project can support several practical use cases:

- **Developers:** discover relevant repositories beyond globally popular projects.
- **Maintainers:** anticipate growth and prepare documentation, support, and infrastructure.
- **Technology teams:** identify emerging tools and changing software ecosystems.
- **Platform designers:** study recommendation quality, novelty, and popularity concentration.
- **Researchers and analysts:** examine temporal growth, collective attention, and feedback loops.
- **Business decision-makers:** support technology scouting and open-source adoption analysis.

The outputs are intended as **decision-support signals**, not as direct measures of software quality.

---

## Project objectives

The main objectives were to:

- build a reproducible GitHub dataset using the GitHub REST API;
- store cleaned and structured data in SQLite;
- perform exploratory analysis and advanced feature engineering;
- create leakage-aware train, validation, and test splits;
- test multiple models for each task;
- use cross-validation, regularization, and hyperparameter tuning where appropriate;
- evaluate models with task-specific metrics;
- freeze selected models before opening the final test sets;
- create separate training and prediction workflows;
- save operational predictions and recommendations back to a database;
- investigate herd behavior and recommendation popularity bias;
- register frozen models and historical metrics in MLflow;
- orchestrate safe validation with Prefect;
- package the system in Docker;
- validate it with a real Kubernetes Job;
- automate clean-checkout validation with GitHub Actions.

---

## Dataset

### Data source

The dataset was collected from the **GitHub REST API**. Repositories were sampled from technology-related topics, including machine learning, data science, deep learning, artificial intelligence, Python, JavaScript, React, web development, large language models, and generative AI.

### Main database tables

The Phase 2 SQLite database is:

```text
database/github_herd.db
```

Its main tables are:

| Table | Purpose | Rows |
|---|---|---:|
| `repositories` | Repository metadata | 763 |
| `stars` | Raw user–repository star events | 216,566 |
| `user_repo_interactions` | Recommendation interaction table | 216,566 |
| `daily_timeseries` | Daily repository star counts | 11,128 |
| `weekly_timeseries` | Weekly repository star counts | 2,250 |
| `herd_modeling` | Repository-level herd modeling data | 145 |

The operational recommendation model was fit using:

- **151,142 users**
- **145 repositories**
- **203,265 train-plus-validation interactions**

### Important processed datasets

```text
data/processed/herd_model_ready.csv
data/processed/weekly_timeseries_features.csv
data/processed/recommendation_features.csv
```

The recommendation delivery package also includes:

```text
data/modeling/recommendation/recommendation_train.csv
data/modeling/recommendation/recommendation_validation.csv
```

### Data limitations

The dataset is affected by GitHub API limits, repository sampling choices, missing or deleted entities, incomplete observation of private activity, and the fact that stars represent attention rather than verified usage or software quality.

---

## Machine-learning tasks

### 1. Four-week popularity forecasting

**Target**

```text
future_4week_stars
```

**Goal:** predict repository star growth over the next 28 days.

The selected model combines a deterministic rolling baseline with a learned residual correction:

```text
Prediction
= 4 × rolling_3week_mean_stars
+ ExtraTrees residual correction
```

The seven selected recent-history features are:

```text
weekly_new_stars
previous_week_stars
weekly_growth_rate
lag_2_week_stars
rolling_3week_mean_stars
rolling_4week_sum_stars
weekly_growth_acceleration
```

### 2. High-growth classification

**Target**

```text
future_growth_surge
```

**Goal:** estimate whether a repository will enter a future high-growth state.

The selected deployment model is a **temporally calibrated logistic-regression classifier**. The decision threshold was selected using validation data and frozen before the final test was opened.

### 3. Herd-aware recommendation

**Goal:** rank unseen repositories for each user and return a top-10 list.

The selected model is:

```text
herd_mitigation_penalty_0.05
```

It uses:

- item–item collaborative similarity: weight `0.75`;
- graph-based personalized PageRank: weight `0.25`;
- candidate-rank normalization;
- popularity penalty: `0.05`;
- seen-item exclusion;
- deterministic repository-ID tie breaking.

The primary selected model does not directly use static content or the forecast score. Content-based, SVD, popularity, and forecast-aware branches were evaluated as development or secondary analyses, but the herd-mitigation reranker was frozen as the final recommender.

---

## Repository structure

A simplified view of the repository is shown below.

```text
.
├── .github/
│   └── workflows/
│       ├── data_pipeline.yml
│       └── phase3_delivery.yml
├── config/
│   └── phase3_config.json
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── modeling/
│       ├── classification/
│       ├── forecasting/
│       └── recommendation/
├── database/
│   ├── github_herd.db
│   └── generated/
├── flows/
│   └── phase3_prefect_flow.py
├── k8s/
│   └── phase3-validation-job.yaml
├── models/
│   ├── classification/
│   ├── forecasting/
│   └── recommendation/
├── outputs/
│   └── phase3/
│       ├── classification/
│       ├── forecasting/
│       ├── recommendation/
│       ├── operational_predictions/
│       ├── mlflow/
│       ├── prefect/
│       └── delivery/
├── scripts/
│   ├── data and database scripts
│   ├── feature-engineering scripts
│   ├── model-training scripts
│   ├── frozen-test evaluation scripts
│   ├── operational prediction scripts
│   └── delivery validation scripts
├── Dockerfile
├── Dockerfile.phase3
├── Dockerfile.phase3.dockerignore
├── pipeline.py
├── prefect.yaml
├── requirements.txt
├── requirements-bonus.txt
├── requirements-bonus-lock.txt
├── requirements-phase3-runtime.txt
├── run_phase3.sh
└── README.md
```

### Directory responsibilities

- `data/`: raw, interim, processed, and task-specific modeling data.
- `database/`: tracked Phase 2 database and ignored generated prediction databases.
- `models/`: frozen classification, forecasting, and recommendation artifacts.
- `scripts/`: data, modeling, evaluation, prediction, audit, and deployment utilities.
- `flows/`: Prefect orchestration logic.
- `outputs/phase3/`: model comparisons, figures, final-test evidence, operational summaries, and deployment evidence.
- `k8s/`: Kubernetes Job specification.
- `.github/workflows/`: CI/CD workflows.
- `run_phase3.sh`: portable, read-only Phase 3 delivery validator.

---

## Pipeline architecture

### Phase 1 and Phase 2 data pipeline

```text
GitHub REST API
      ↓
Raw CSV files
      ↓
Cleaning and validation
      ↓
SQLite database
      ↓
Exploratory analysis
      ↓
Preprocessing
      ↓
Feature engineering
      ↓
Task-specific modeling datasets
```

The original data pipeline can be run with:

```bash
python pipeline.py
```

It executes the database import, SQL validation, data loading, exploratory analysis, preprocessing, and feature-engineering stages in sequence.

### Training workflow

```text
Database and processed data
      ↓
Leakage-aware task dataset construction
      ↓
Train / validation / test separation
      ↓
Cross-validation and hyperparameter search
      ↓
External validation comparison
      ↓
Model and threshold selection
      ↓
Pre-test artifact freeze
      ↓
One-time final test evaluation
      ↓
Frozen deployment artifact and evidence
```

### Prediction workflow

```text
Latest repository and interaction data
      ↓
Apply the same preprocessing and feature contracts
      ↓
Load frozen classification model
      ↓
Load frozen forecasting model
      ↓
Load frozen recommendation model
      ↓
Generate repository predictions and top-10 recommendations
      ↓
Save results to a generated SQLite database
      ↓
Write compact summaries and manifests
```

The prediction workflow never retrains models on prediction data.

---

## Environment setup

Python `3.12` is recommended.

### Main modeling environment

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main modeling environment was validated with versions including:

```text
Python          3.12.10
NumPy           2.5.1
pandas          3.0.3
scikit-learn    1.9.0
joblib          1.5.3
XGBoost         3.3.0
```

### Bonus MLOps environment

MLflow and Prefect are isolated in a separate environment because their compatible dependency set uses pandas `2.3.3`.

```bash
python3.12 -m venv venv_bonus
source venv_bonus/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-bonus-lock.txt
```

Validated bonus versions include:

```text
MLflow          3.14.0
Prefect         3.7.8
pandas          2.3.3
scikit-learn    1.9.0
```

### Minimal delivery environment

For portable validation without the complete development stack:

```bash
python3.12 -m venv venv_runtime
source venv_runtime/bin/activate
python -m pip install -r requirements-phase3-runtime.txt
```

---

## How to run the project

### Recommended safe validation command

From the project root:

```bash
./run_phase3.sh --batch-size 5000
```

This command performs a **read-only delivery validation**. It:

- verifies the frozen model artifacts and known hashes;
- verifies committed MLflow and Prefect evidence;
- validates operational classification, forecasting, and recommendation inference;
- confirms that no protected model or database file changes;
- does not retrain models;
- does not rerun final-test evaluation;
- does not create new operational predictions.

A successful run ends with:

```text
Validation status: passed_no_files_written
Delivery validation status: PASSED
```

### Direct portable validator

```bash
python scripts/validate_phase3_delivery.py --batch-size 5000
```

### Validate operational inference only

```bash
python scripts/generate_operational_predictions.py   --run-id delivery_validation   --batch-size 5000   --validate-only
```

### Generate a new operational prediction run

This command writes to a separate generated database and output directory:

```bash
python scripts/generate_operational_predictions.py   --database-path database/generated/github_herd_phase3_predictions.db   --output-root outputs/phase3/operational_predictions   --run-id operational_<unique_id>   --batch-size 5000
```

Requirements for a write-enabled run:

- use a unique run ID;
- do not point `--database-path` to the tracked `database/github_herd.db`;
- do not overwrite an existing operational output directory;
- keep generated databases under `database/generated/`.

### Development and training scripts

The repository retains the development scripts used for model search, tuning, auditing, and freezing. Important examples include:

```text
scripts/train_corrected_classification_models.py
scripts/classification_temporal_cv.py
scripts/train_forecasting_models.py
scripts/audit_forecasting_baseline_residual_models.py
scripts/train_recommendation_models.py
scripts/audit_recommendation_model_search.py
```

> **Scientific freeze warning:** the final models and test results are already frozen. Do not rerun final-test scripts or replace the selected artifacts when reproducing the submitted result. Training scripts are retained for transparency and development reproduction.

---

## Model development and scientific protocol

The project followed a strict model-selection protocol:

1. Create task-specific train, validation, and test datasets.
2. Protect temporal ordering and avoid future-information leakage.
3. Compare multiple model families.
4. Tune hyperparameters using training data and temporal cross-validation.
5. Compare one selected candidate from each family on external validation data.
6. Select model parameters and decision thresholds without using the test set.
7. Freeze model artifacts, metadata, input hashes, and Git references.
8. Open each test set once for final evaluation.
9. Record final metrics and prohibit post-test model changes.
10. Use frozen artifacts for operational inference and deployment.

This process reduces optimistic bias and prevents repeated adaptation to the final test data.

---

## Final model results

All metrics below are from the frozen one-time test evaluations.

### Forecasting

**Selected approach:** additive residual Extra Trees<br>
**Test rows:** 66<br>
**Test repositories:** 22<br>
**Forecast horizon:** 28 days

| Metric | Frozen test result |
|---|---:|
| MAE | 127.0481 |
| RMSE | 277.9668 |
| RMSLE | 0.9174 |
| sMAPE | 0.5038 |
| Median absolute error | 27.2615 |
| Maximum absolute error | 1,208.8551 |
| Forecast bias | -88.8777 |

The negative forecast bias indicates that the model tended to **underpredict future growth**, especially for unusually large growth events. The difference between the median error and maximum error also shows that a small number of extreme repositories dominate the aggregate error.

### High-growth classification

**Selected approach:** temporally calibrated logistic regression<br>
**Frozen threshold:** `0.1748254577`<br>
**Test rows:** 66<br>
**Test repositories:** 22

| Metric | Frozen test result |
|---|---:|
| Accuracy | 0.6061 |
| Balanced accuracy | 0.5810 |
| Precision | 0.3333 |
| Recall | 0.5294 |
| F1 score | 0.4091 |
| ROC-AUC | 0.6483 |
| PR-AUC | 0.3983 |
| MCC | 0.1441 |
| Brier score | 0.1890 |
| Log loss | 0.5705 |

Confusion matrix counts:

```text
True negatives: 31
False positives: 18
False negatives: 8
True positives: 9
```

The classifier provides moderate ranking information but is not sufficiently accurate for fully automated high-stakes decisions. Its most appropriate use is as an **early-warning or prioritization signal** that supports human review.

### Recommendation

**Selected model:** herd-mitigation reranker<br>
**Evaluation users:** 13,301<br>
**Catalog:** 145 repositories<br>
**Top-K:** 10

| Metric | Frozen test result |
|---|---:|
| Precision@10 | 0.02227 |
| Recall@10 | 0.22269 |
| Hit Rate@10 | 0.22269 |
| NDCG@10 | 0.11779 |
| MAP@10 | 0.08638 |
| MRR | 0.10923 |
| Catalog Coverage@10 | 0.92414 |
| Long-tail share@10 | 0.86851 |
| Intra-list diversity@10 | 0.95312 |
| Novelty@10 | 7.18350 |
| Mean positive rank | 43.7935 |
| Median positive rank | 33 |

The ranking metrics reflect a difficult sparse recommendation problem with one held-out positive among a large candidate set. At the same time, the recommender achieved high catalog coverage, long-tail exposure, and intra-list diversity. This indicates that it avoids collapsing recommendations onto only a small group of globally popular repositories.

---

## Operational predictions

The frozen operational run is:

```text
operational_v1_1cafd65
```

Its validated outputs include:

| Operational quantity | Value |
|---|---:|
| Repository prediction rows | 82 |
| Predicted growth surges | 12 |
| Mean four-week forecast | 945.6431 |
| Recommendation users | 151,142 |
| Recommendation catalog | 145 |
| Recommendations per user | 10 |
| Expected recommendation rows | 1,511,420 |

Compact operational outputs are stored under:

```text
outputs/phase3/operational_predictions/operational_v1_1cafd65/
```

Important files include:

```text
operational_run_manifest.json
operational_run_summary.json
repository_predictions.csv
recommendation_repository_frequency.csv
recommendation_sample_first_100_users.csv
```

The complete generated prediction database is intentionally ignored by Git because of its size:

```text
database/generated/github_herd_phase3_predictions.db
```

---

## Database integration

The assignment requires final predictions to be saved back to a database. The operational prediction pipeline writes to a separate generated SQLite database and preserves the tracked Phase 2 database unchanged.

The generated database contains operational tables for:

- model run metadata;
- repository-level four-week forecasts;
- high-growth probabilities and labels;
- user-level ranked recommendations.

Important validation rules include:

- repository prediction row counts must match the operational manifest;
- recommendation rows must equal `users × top_k`;
- each user–rank pair must be unique;
- previously seen repositories must be excluded;
- the tracked Phase 2 database hash must not change.

---

## Prefect orchestration

The Prefect flow is:

```text
flows/phase3_prefect_flow.py
```

The deployment configuration is:

```text
prefect.yaml
```

The flow name is:

```text
github-herd-phase3-safe-orchestration
```

Its default mode is read-only. It:

1. verifies frozen classification, forecasting, and recommendation artifacts;
2. validates existing MLflow registration evidence;
3. validates operational inference;
4. refuses prediction writes unless explicitly authorized.

### Run the flow directly

```bash
source venv_bonus/bin/activate

PREFECT_SERVER_ANALYTICS_ENABLED=false DO_NOT_TRACK=1 python flows/phase3_prefect_flow.py   --batch-size 5000
```

### Write-enabled Prefect execution

A write-enabled run requires both an explicit flag and a run ID beginning with `prefect_`:

```bash
python flows/phase3_prefect_flow.py   --batch-size 5000   --execute-operational-predictions   --run-id prefect_<unique_id>   --database-path database/generated/github_herd_phase3_prefect_predictions.db   --output-root outputs/phase3/operational_predictions
```

The project also validated:

- direct flow execution;
- deployment registration;
- a local Process work pool;
- a real worker-triggered deployment run;
- final Prefect state `COMPLETED`.

Evidence is stored under:

```text
outputs/phase3/prefect/
```

---

## MLflow model management

MLflow is used to register already-frozen models and their historical metrics without retraining or reopening final-test decisions.

The registration script is:

```text
scripts/register_frozen_models_mlflow.py
```

### Validate MLflow inputs without writing

```bash
source venv_bonus/bin/activate

python scripts/register_frozen_models_mlflow.py   --validate-only
```

### Register the frozen models locally

```bash
python scripts/register_frozen_models_mlflow.py
```

The local tracking directory is:

```text
.mlflow_local/
```

The experiment is:

```text
github-herd-phase3-frozen-models
```

Registered components include:

- classification model;
- forecasting model;
- recommendation model;
- operational prediction contract.

The selected model versions use the alias:

```text
champion
```

Portable MLflow evidence is stored in:

```text
outputs/phase3/mlflow/frozen_registry_summary.json
```

---

## Docker, Kubernetes, and CI/CD

### Docker

Build the Phase 3 validation image:

```bash
docker build   --file Dockerfile.phase3   --tag github-herd-phase3:latest   .
```

Run the hardened container:

```bash
docker run   --rm   --user 10001:10001   --network none   --read-only   --tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m   --cap-drop ALL   --security-opt no-new-privileges:true   --pids-limit 256   --cpus 2   --memory 2g   github-herd-phase3:latest   --batch-size 5000
```

The Phase 3 image:

- uses Python 3.12;
- runs as a non-root user;
- contains only the required runtime closure;
- excludes the generated predictions database;
- executes the read-only validator by default.

### Kubernetes

Apply the validation Job:

```bash
kubectl apply   --filename k8s/phase3-validation-job.yaml
```

Wait for completion:

```bash
kubectl wait   --for=condition=complete   job/github-herd-phase3-validation   --timeout=10m
```

Read the logs:

```bash
kubectl logs   job/github-herd-phase3-validation
```

The validated Job configuration includes:

- non-root user and group `10001`;
- read-only root filesystem;
- privilege escalation disabled;
- all Linux capabilities dropped;
- service-account token disabled;
- CPU and memory requests and limits;
- temporary writable storage mounted only at `/tmp`;
- no retries after a failed execution (`backoffLimit: 0`).

A real local Kubernetes Job completed successfully with one succeeded Pod, zero restarts, and no warning events.

### GitHub Actions

The Phase 3 CI/CD workflow is:

```text
.github/workflows/phase3_delivery.yml
```

It runs three independent jobs:

1. **Portable read-only validation**
2. **Build and run Phase 3 container**
3. **Validate Kubernetes manifest**

The workflow was validated twice on GitHub Actions. Both runs completed successfully for the final delivery sequence.

Portable CI evidence is stored in:

```text
outputs/phase3/delivery/github_actions_validation_v1.json
```

---

## Reproducibility and safety

The project includes several safeguards:

- model artifacts are frozen and SHA-256 hashed;
- train, validation, and test data contracts are recorded;
- final test sets were opened only for one-time evaluation;
- final-test outputs prohibit post-test model changes;
- operational validation is read-only by default;
- new prediction databases must be written under `database/generated/`;
- the tracked Phase 2 database is protected from modification;
- output directories cannot be silently overwritten;
- Docker and Kubernetes run as non-root;
- protected files are checked before and after delivery validation;
- GitHub Actions validates the project from a clean repository checkout.

### Frozen model hashes

| Component | SHA-256 |
|---|---|
| Classification | `7f66e845d0689c2c29583ec28d93218a602b951a3246a56767646d2cc48f5f23` |
| Forecasting | `e728866f0b523a3393d1fe00cbf58b7500a02181866b337a4c1b78f8d2806294` |
| Recommendation | `f2b178d1c888a04a5dbf15253e0d99b1a920e9230c549c8213e4f529f4c5e878` |
| Phase 2 database | `559803caa16b7f43a55cf6947f4eb580e92bb5a0ed7cd68ad0b581b960f88ee4` |

### Evidence directories

```text
outputs/phase3/classification/
outputs/phase3/forecasting/
outputs/phase3/recommendation/
outputs/phase3/mlflow/
outputs/phase3/prefect/
outputs/phase3/delivery/
```

---

## Real-world interpretation

### Forecasting

The model can identify repositories expected to receive substantial near-term attention. This may help maintainers anticipate support demand and help organizations monitor emerging technologies. However, the negative test bias and large maximum error show that sudden viral events remain difficult to predict.

### High-growth classification

The classifier can be used to create an early-warning shortlist. Its moderate ROC-AUC and recall make it useful for prioritization, but its false-positive rate means it should not be used as an automatic adoption or investment rule.

### Recommendation

The recommender recovers approximately 22.3% of held-out repositories in the top 10 while exposing more than 92% of the catalog and maintaining a high long-tail share. From a platform perspective, this creates a useful relevance–diversity trade-off and reduces the risk that recommendation traffic is concentrated only on already-famous repositories.

### Business and platform implications

Potential applications include:

- technology trend dashboards;
- open-source ecosystem monitoring;
- personalized developer discovery;
- maintainer capacity planning;
- long-tail repository exposure;
- internal technology scouting.

These applications require periodic retraining, drift monitoring, manipulation detection, and human review.

---

## Limitations

### Data limitations

- sampled rather than complete GitHub ecosystem;
- GitHub API rate and pagination constraints;
- incomplete observation of private activity;
- missing or deleted entities;
- stars are not equivalent to installation, retention, or quality.

### Modeling limitations

- only 145 repositories in the main recommendation catalog;
- only 22 repositories represented in the final repository-level test evaluations;
- rare viral growth events dominate forecasting error;
- classification performance is moderate;
- recommendation evaluation is offline;
- cold-start users and repositories remain difficult;
- external events and social-media exposure are not modeled directly.

### Deployment limitations

- SQLite is suitable for the project but not a distributed production datastore;
- Prefect and MLflow were validated locally;
- the Kubernetes artifact is a batch validation Job rather than an online service;
- no production API, online A/B test, or live drift-monitoring dashboard is included.

---

## Presentation and report

The final presentation must cover Phases 1–3 and remain within the course time limit.

Replace the placeholders below before submission:

```text
Final report: [report.pdf](report.pdf)
PowerPoint:    To be added
Video:         To be added
```

The links may also be placed in:

```text
video_link.txt
```

---

## Validated Git references

### Delivery infrastructure

```text
Commit: d8f89bf941bcffe5a65738beef6a91155fe8e5e5
Tag:    phase3-delivery-validation-v1
```

### CI evidence and final synchronized branch

```text
Commit: 8a864558bc8ea79543a88bd587948cda3a3df087
Tag:    phase3-ci-evidence-v1
Branch: phase3
```

### GitHub Actions

The Phase 3 workflow completed successfully for:

```text
Run 1: delivery infrastructure commit
Run 2: CI evidence commit
```

---

## Responsible use

Repository popularity and model predictions should not be interpreted as measures of intrinsic software quality. Rankings may influence future attention and therefore create feedback loops. The outputs should be combined with code quality, maintenance activity, security, licensing, documentation, and stakeholder-specific requirements before making real-world decisions.
