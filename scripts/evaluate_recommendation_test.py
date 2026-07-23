import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data/modeling/recommendation/recommendation_train.csv'
VALID = ROOT / 'data/modeling/recommendation/recommendation_validation.csv'
TEST = ROOT / 'data/modeling/recommendation/recommendation_test.csv'
TRAIN_SCRIPT = ROOT / 'scripts/train_recommendation_models.py'
ARTIFACT = ROOT / 'models/recommendation/selected_recommender_pretest.joblib'
METADATA = ROOT / 'models/recommendation/selected_recommender_pretest_metadata.json'
PRETEST_MANIFEST = ROOT / 'outputs/phase3/recommendation/pretest_freeze/recommendation_pretest_freeze_manifest.json'
DESIGN_MANIFEST = ROOT / 'outputs/phase3/recommendation/design_freeze/recommendation_design_freeze_manifest.json'
OUTPUT = ROOT / 'outputs/phase3/recommendation/final_test'
TAG = 'recommendation-pretest-freeze-v1'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def load_module():
    spec = importlib.util.spec_from_file_location('recommendation_test_module', TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not import training functions.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_freeze():
    if OUTPUT.exists():
        raise RuntimeError(f'Refusing to overwrite {OUTPUT.relative_to(ROOT)}')
    head = git('rev-parse', 'HEAD')
    target = git('rev-list', '-n', '1', TAG)
    if git('branch', '--show-current') != 'phase3' or head != target:
        raise RuntimeError('HEAD must equal the recommendation pretest tag.')

    artifact = joblib.load(ARTIFACT)
    metadata = read_json(METADATA)
    manifest = read_json(PRETEST_MANIFEST)
    artifact_hash = sha256(ARTIFACT)
    if not (artifact_hash == metadata['artifact']['sha256'] == manifest['artifact_sha256']):
        raise RuntimeError('Pretest artifact hashes do not match.')
    if artifact['artifact_version'] != 'recommendation_pretest_v1':
        raise RuntimeError('Unexpected pretest artifact version.')
    if artifact['test_data_status'] != 'header_only_not_loaded_or_evaluated':
        raise RuntimeError('Pretest test status is not sealed.')
    if float(artifact['scoring_contract']['forecast_weight']) != 0.0:
        raise RuntimeError('Forecast weight is not zero.')
    if bool(artifact['scoring_contract']['static_content_in_primary']):
        raise RuntimeError('Static content is unexpectedly enabled.')

    for relative, expected in manifest['input_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f'Pretest input hash mismatch: {relative}')
    design = read_json(DESIGN_MANIFEST)
    test_relative = str(TEST.relative_to(ROOT))
    if manifest['input_hashes'][test_relative] != design['input_hashes'][test_relative]:
        raise RuntimeError('Test hash changed between freezes.')
    return artifact, artifact_hash, head, manifest['input_hashes'][test_relative]


def score_model(module, context, artifact):
    winners = artifact['internal_family_winners']
    components = artifact['component_artifacts']
    item_scores = module.item_item_scores(
        context,
        np.asarray(components['item_similarity'], dtype=float),
        winners['item_item_cosine']['parameters']['aggregation'],
    )
    svd_components = np.asarray(components['svd_components'], dtype=float)
    svd_scores = np.asarray(
        (context.history_matrix @ svd_components.T) @ svd_components,
        dtype=float,
    )
    graph_scores = module.graph_scores(
        context,
        np.asarray(components['graph_propagation'], dtype=float),
    )
    popularity_scores = module.popularity_scores(context)
    raw = {
        'item_item_cosine': item_scores,
        'truncated_svd': svd_scores,
        'graph_personalized_pagerank': graph_scores,
        'training_popularity': popularity_scores,
    }
    normalized = {
        name: module.rank_normalize_scores(scores, context.seen_mask)
        for name, scores in raw.items()
    }
    hybrid = np.zeros_like(item_scores, dtype=float)
    for name, weight in artifact['hybrid_parameters']['weights'].items():
        hybrid += float(weight) * normalized[name]
    hybrid_rank = module.rank_normalize_scores(hybrid, context.seen_mask)
    final_scores = hybrid_rank - float(
        artifact['herd_parameters']['popularity_penalty']
    ) * normalized['training_popularity']
    if not np.isfinite(final_scores).all():
        raise RuntimeError('Final scores contain nonfinite values.')
    return final_scores


def build_top10(module, scores, context):
    masked = module.apply_candidate_mask(scores, context.seen_mask)
    order = module.stable_top_order(masked)[:, :10]
    rows = []
    for user_row, user_id in enumerate(context.eval_user_ids):
        for rank, repo_index in enumerate(order[user_row], start=1):
            rows.append({
                'user_id': int(user_id),
                'rank': rank,
                'repo_id': int(context.repo_ids[repo_index]),
                'repo_full_name': str(context.repo_names[repo_index]),
                'score': float(scores[user_row, repo_index]),
                'train_validation_count': float(context.item_counts[repo_index]),
            })
    return pd.DataFrame(rows)


def temporal_summary(combined, test):
    latest = combined.groupby('user_id')['starred_at'].max().rename('latest_history')
    frame = test[['user_id', 'starred_at']].merge(
        latest, on='user_id', how='left', validate='one_to_one'
    )
    return {
        'test_users': int(len(frame)),
        'strictly_later_than_latest_history': int((frame['starred_at'] > frame['latest_history']).sum()),
        'equal_to_latest_history': int((frame['starred_at'] == frame['latest_history']).sum()),
        'earlier_than_latest_history': int((frame['starred_at'] < frame['latest_history']).sum()),
        'policy': 'all frozen test rows evaluated; diagnostics reported without post-test filtering',
    }


def save_plots(per_user, top10):
    fig, ax = plt.subplots(figsize=(10, 6))
    maximum = int(per_user['positive_rank'].max())
    ax.hist(per_user['positive_rank'], bins=np.arange(0.5, maximum + 1.5, 1.0))
    ax.set_xlabel('Positive repository rank')
    ax.set_ylabel('Users')
    ax.set_title('Final Test Positive-Rank Distribution')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'recommendation_final_test_positive_rank_distribution.png',
                dpi=180, bbox_inches='tight')
    plt.close(fig)

    frequency = (
        top10.groupby(['repo_id', 'repo_full_name']).size().rename('count').reset_index()
        .sort_values(['count', 'repo_id'], ascending=[False, True]).head(20)
        .sort_values('count')
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(frequency['repo_full_name'], frequency['count'])
    ax.set_xlabel('Number of top-10 appearances')
    ax.set_title('Most Frequently Recommended Repositories on Final Test')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'recommendation_final_test_top_repository_frequency.png',
                dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    artifact, artifact_hash, pretest_commit, test_hash = verify_freeze()
    module = load_module()
    train = module.load_interactions(TRAIN)
    validation = module.load_interactions(VALID)

    # The one and only full read of the frozen recommendation test split.
    test = module.load_interactions(TEST)

    if test['user_id'].duplicated().any():
        raise RuntimeError('Test must contain one positive repository per user.')
    combined = pd.concat([train, validation], ignore_index=True)
    if combined.duplicated(['user_id', 'repo_id']).any():
        raise RuntimeError('Train plus validation contains duplicate pairs.')
    overlap = combined[['user_id', 'repo_id']].merge(
        test[['user_id', 'repo_id']], on=['user_id', 'repo_id'], how='inner'
    )
    if len(overlap):
        raise RuntimeError('Train/validation and test pairs overlap.')

    catalog = pd.DataFrame({
        'repo_id': artifact['catalog_repo_ids'],
        'repo_full_name': artifact['catalog_repo_names'],
    }).sort_values('repo_id').reset_index(drop=True)
    if len(catalog) != 145:
        raise RuntimeError('Unexpected catalog size.')
    if set(test['repo_id']) - set(catalog['repo_id']):
        raise RuntimeError('Test contains repositories outside the catalog.')
    if set(test['user_id']) - set(combined['user_id']):
        raise RuntimeError('Test contains users without prior history.')

    context = module.build_context(combined, test, catalog, 'repo_id')
    if not np.array_equal(
        context.repo_ids,
        np.asarray(artifact['catalog_repo_ids'], dtype=np.int64),
    ):
        raise RuntimeError('Catalog order differs from the frozen artifact.')
    if not np.array_equal(
        context.item_counts,
        np.asarray(artifact['item_counts'], dtype=float),
    ):
        raise RuntimeError('Item counts differ from the frozen artifact.')

    scores = score_model(module, context, artifact)
    metrics, per_user = module.evaluate_scores(
        scores=scores,
        context=context,
        model_family=artifact['selected_family'],
        model_name=artifact['selected_model_name'],
        parameters=artifact['selected_parameters'],
        return_user_ranks=True,
        compute_beyond_accuracy=True,
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    metrics_path = OUTPUT / 'recommendation_final_test_metrics.csv'
    ranks_path = OUTPUT / 'recommendation_final_test_per_user_ranks.csv'
    top10_path = OUTPUT / 'recommendation_final_test_top10.csv.gz'
    summary_path = OUTPUT / 'recommendation_final_test_summary.json'

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    per_user.to_csv(ranks_path, index=False)
    top10 = build_top10(module, scores, context)
    top10.to_csv(top10_path, index=False, compression='gzip')
    save_plots(per_user, top10)

    candidate_counts = (~context.seen_mask).sum(axis=1)
    temporal = temporal_summary(combined, test)
    summary = {
        'status': 'recommendation_final_test_evaluated_once',
        'evaluation_policy': (
            'one-time test evaluation; no tuning or model changes after viewing results'
        ),
        'pretest_commit': pretest_commit,
        'pretest_tag': TAG,
        'artifact_version': artifact['artifact_version'],
        'artifact_sha256': artifact_hash,
        'selected_family': artifact['selected_family'],
        'selected_model_name': artifact['selected_model_name'],
        'selected_parameters': artifact['selected_parameters'],
        'hybrid_parameters': artifact['hybrid_parameters'],
        'herd_parameters': artifact['herd_parameters'],
        'train_rows': int(len(train)),
        'validation_rows': int(len(validation)),
        'test_rows': int(len(test)),
        'test_users': int(test['user_id'].nunique()),
        'test_repositories': int(test['repo_id'].nunique()),
        'catalog_repositories': int(len(catalog)),
        'test_file_sha256': test_hash,
        'candidate_count': {
            'minimum': int(candidate_counts.min()),
            'median': float(np.median(candidate_counts)),
            'mean': float(candidate_counts.mean()),
            'maximum': int(candidate_counts.max()),
        },
        'temporal_diagnostics': temporal,
        'metrics': {
            key: float(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else value
            for key, value in metrics.items()
        },
        'top10_rows': int(len(top10)),
        'outputs': {
            'metrics': str(metrics_path.relative_to(ROOT)),
            'per_user_ranks': str(ranks_path.relative_to(ROOT)),
            'top10_recommendations': str(top10_path.relative_to(ROOT)),
            'positive_rank_plot': str(
                (OUTPUT / 'recommendation_final_test_positive_rank_distribution.png').relative_to(ROOT)
            ),
            'top_repository_plot': str(
                (OUTPUT / 'recommendation_final_test_top_repository_frequency.png').relative_to(ROOT)
            ),
        },
        'post_test_status': 'final_results_frozen_no_retuning',
    }
    write_json(summary_path, summary)

    print('=' * 108)
    print('FINAL RECOMMENDATION TEST')
    print('=' * 108)
    print('Artifact version:', artifact['artifact_version'])
    print('Selected model:', artifact['selected_model_name'])
    print('Test rows/users:', len(test))
    print('Test repositories:', test['repo_id'].nunique())
    print()
    print(pd.DataFrame([metrics]).to_string(index=False))
    print()
    print('Temporal diagnostics:', temporal)
    print(
        'Candidate count min/median/mean/max:',
        int(candidate_counts.min()),
        float(np.median(candidate_counts)),
        float(candidate_counts.mean()),
        int(candidate_counts.max()),
    )
    print('Top-10 rows:', len(top10))
    print('Final status: final_results_frozen_no_retuning')


if __name__ == '__main__':
    main()
