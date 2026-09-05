import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data/modeling/recommendation/recommendation_train.csv'
VALID = ROOT / 'data/modeling/recommendation/recommendation_validation.csv'
TEST = ROOT / 'data/modeling/recommendation/recommendation_test.csv'
TRAIN_SCRIPT = ROOT / 'scripts/train_recommendation_models.py'
DESIGN_MANIFEST = ROOT / 'outputs/phase3/recommendation/design_freeze/recommendation_design_freeze_manifest.json'
DESIGN_METADATA = ROOT / 'outputs/phase3/recommendation/design_freeze/recommendation_design_freeze_metadata.json'
DEV_ARTIFACT = ROOT / 'models/recommendation/selected_recommender_development.joblib'
DEV_METADATA = ROOT / 'models/recommendation/selected_recommender_development_metadata.json'
SELECTION_SUMMARY = ROOT / 'outputs/phase3/recommendation/model_search/recommendation_model_selection_summary.json'
SECONDARY_SUMMARY = ROOT / 'outputs/phase3/recommendation/secondary_branches/secondary_recommendation_summary.json'
PRETEST_ARTIFACT = ROOT / 'models/recommendation/selected_recommender_pretest.joblib'
PRETEST_METADATA = ROOT / 'models/recommendation/selected_recommender_pretest_metadata.json'
OUTPUT_DIR = ROOT / 'outputs/phase3/recommendation/pretest_freeze'
REFIT_SUMMARY = OUTPUT_DIR / 'recommendation_pretest_refit_summary.json'
MANIFEST = OUTPUT_DIR / 'recommendation_pretest_freeze_manifest.json'
PRETEST_TAG = 'recommendation-pretest-freeze-v1'
PRIMARY_TAG = 'recommendation-primary-selection-v1'
SECONDARY_TAG = 'recommendation-secondary-analysis-v1'
DESIGN_TAG = 'recommendation-design-freeze-v1'


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


def git(*arguments):
    return subprocess.run(['git', *arguments], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def tag_target(name):
    return git('rev-list', '-n', '1', name)


def load_training_module():
    spec = importlib.util.spec_from_file_location('recommendation_training_pretest', TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not import recommendation training code.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_start_state():
    existing = [str(p.relative_to(ROOT)) for p in
                [PRETEST_ARTIFACT, PRETEST_METADATA, OUTPUT_DIR] if p.exists()]
    if existing:
        raise RuntimeError('Refusing to overwrite existing pretest outputs:\n' + '\n'.join(existing))
    branch = git('branch', '--show-current')
    head = git('rev-parse', 'HEAD')
    primary = tag_target(PRIMARY_TAG)
    secondary = tag_target(SECONDARY_TAG)
    design = tag_target(DESIGN_TAG)
    if branch != 'phase3':
        raise RuntimeError(f'Expected phase3 branch, found {branch}.')
    if head != secondary:
        raise RuntimeError('HEAD must equal recommendation-secondary-analysis-v1.')
    header = pd.read_csv(TEST, nrows=0)
    required = {'user_id', 'repo_id', 'repo_full_name', 'starred_at'}
    missing = required - set(header.columns)
    if missing:
        raise RuntimeError(f'Test header missing columns: {sorted(missing)}')
    if len(header) != 0:
        raise RuntimeError('Header-only test read unexpectedly loaded rows.')
    return {'branch': branch, 'head': head,
            'primary_selection_commit': primary,
            'secondary_analysis_commit': secondary,
            'design_freeze_commit': design,
            'test_header_columns': list(header.columns)}


def verify_design_hashes(manifest):
    verified = {}
    expected = {**manifest['input_hashes'], **manifest['artifact_hashes']}
    for relative, expected_hash in expected.items():
        actual = sha256(ROOT / relative)
        if actual != expected_hash:
            raise RuntimeError(f'Design-freeze hash mismatch: {relative}')
        verified[relative] = actual
    return verified


def load_combined(module):
    train = module.load_interactions(TRAIN)
    validation = module.load_interactions(VALID)
    combined = pd.concat([train, validation], ignore_index=True)
    if len(combined) != len(train) + len(validation):
        raise RuntimeError('Combined interaction row count is inconsistent.')
    if combined.duplicated(['user_id', 'repo_id']).any():
        raise RuntimeError('Train plus validation contains duplicate user-repository pairs.')
    overlap = train[['user_id', 'repo_id']].merge(
        validation[['user_id', 'repo_id']], on=['user_id', 'repo_id'], how='inner')
    if len(overlap):
        raise RuntimeError('Train and validation user-repository pairs overlap.')
    return train, validation, combined


def fit_components(module, combined, catalog, dev):
    repo_ids = catalog['repo_id'].to_numpy(dtype=np.int64)
    repo_names = catalog['repo_full_name'].astype(str).to_numpy()
    repo_to_index = {int(repo_id): index for index, repo_id in enumerate(repo_ids)}
    user_ids = np.sort(combined['user_id'].unique().astype(np.int64))
    matrix = module.build_sparse_matrix(combined, user_ids, repo_to_index)
    if matrix.shape != (len(user_ids), len(repo_ids)):
        raise RuntimeError('Unexpected combined sparse-matrix shape.')
    if matrix.nnz != len(combined):
        raise RuntimeError('Sparse-matrix nnz does not match combined rows.')

    context = SimpleNamespace(fit_matrix_all_users=matrix)
    winners = dev['internal_family_winners']
    item_params = winners['item_item_cosine']['parameters']
    item_similarity = module.build_item_similarity(
        context, float(item_params['shrinkage']))

    graph_params = winners['graph_personalized_pagerank']['parameters']
    if graph_params['edge_weight'] == 'ppmi':
        edge_matrix = module.build_ppmi_matrix(context)
    elif graph_params['edge_weight'] == 'cosine':
        edge_matrix = item_similarity
    else:
        raise RuntimeError(f"Unsupported graph edge weight: {graph_params['edge_weight']}")
    graph_propagation = module.graph_propagation_matrix(
        edge_matrix, float(graph_params['restart_probability']))

    svd_params = winners['truncated_svd']['parameters']
    svd = TruncatedSVD(n_components=int(svd_params['n_components']),
                       n_iter=int(svd_params['n_iter']),
                       random_state=int(module.RANDOM_STATE))
    svd.fit(matrix)
    item_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)

    checks = {
        'fit_matrix_shape': [int(matrix.shape[0]), int(matrix.shape[1])],
        'fit_matrix_nonzero': int(matrix.nnz),
        'item_count_sum': float(item_counts.sum()),
        'minimum_item_count': float(item_counts.min()),
        'maximum_item_count': float(item_counts.max()),
        'item_similarity_shape': list(item_similarity.shape),
        'item_similarity_finite': bool(np.isfinite(item_similarity).all()),
        'item_similarity_symmetric': bool(np.allclose(
            item_similarity, item_similarity.T, atol=1e-12, rtol=0)),
        'item_similarity_zero_diagonal': bool(np.allclose(
            np.diag(item_similarity), 0.0, atol=1e-12, rtol=0)),
        'graph_propagation_shape': list(graph_propagation.shape),
        'graph_propagation_finite': bool(np.isfinite(graph_propagation).all()),
        'svd_components_shape': list(svd.components_.shape),
        'svd_components_finite': bool(np.isfinite(svd.components_).all()),
        'svd_variance_finite': bool(np.isfinite(svd.explained_variance_ratio_).all()),
    }
    required = [checks['item_similarity_finite'], checks['item_similarity_symmetric'],
                checks['item_similarity_zero_diagonal'], checks['graph_propagation_finite'],
                checks['svd_components_finite'], checks['svd_variance_finite'],
                checks['item_count_sum'] == float(len(combined)),
                checks['minimum_item_count'] > 0.0]
    if not all(required):
        raise RuntimeError('Refit component validation failed.')

    return {
        'catalog_repo_ids': repo_ids,
        'catalog_repo_names': repo_names,
        'fit_user_ids': user_ids,
        'item_counts': item_counts,
        'component_artifacts': {
            'item_similarity': item_similarity,
            'svd_components': svd.components_,
            'svd_explained_variance_ratio': svd.explained_variance_ratio_,
            'graph_propagation': graph_propagation,
        },
    }, checks


def verify_reload(original, reloaded):
    for field in ['artifact_version', 'freeze_status', 'selected_family',
                  'selected_model_name', 'test_data_status']:
        if original[field] != reloaded[field]:
            raise RuntimeError(f'Reload mismatch: {field}')
    differences = []
    for field in ['catalog_repo_ids', 'catalog_repo_names', 'fit_user_ids', 'item_counts']:
        left, right = np.asarray(original[field]), np.asarray(reloaded[field])
        if left.dtype.kind in {'U', 'S', 'O'}:
            if not np.array_equal(left, right):
                raise RuntimeError(f'Reload array mismatch: {field}')
            differences.append(0.0)
        else:
            differences.append(float(np.max(np.abs(left.astype(float) - right.astype(float)))))
    for field in ['item_similarity', 'svd_components',
                  'svd_explained_variance_ratio', 'graph_propagation']:
        left = np.asarray(original['component_artifacts'][field], dtype=float)
        right = np.asarray(reloaded['component_artifacts'][field], dtype=float)
        differences.append(float(np.max(np.abs(left - right))))
    maximum = max(differences)
    if maximum != 0.0:
        raise RuntimeError('Reloaded pretest artifact differs from saved artifact.')
    return maximum


def main():
    state = verify_start_state()
    module = load_training_module()
    design_manifest = read_json(DESIGN_MANIFEST)
    if design_manifest['test_data_status'] != 'header_only_not_loaded_or_evaluated':
        raise RuntimeError('Design manifest does not preserve sealed test status.')
    verified_design_hashes = verify_design_hashes(design_manifest)

    dev = joblib.load(DEV_ARTIFACT)
    dev_meta = read_json(DEV_METADATA)
    dev_hash = sha256(DEV_ARTIFACT)
    if dev_hash != dev_meta['development_artifact_sha256']:
        raise RuntimeError('Development artifact hash mismatch.')
    if dev['artifact_version'] != 'recommendation_development_v1':
        raise RuntimeError('Unexpected development artifact version.')
    if dev['test_data_status'] != 'header_only_not_loaded_or_evaluated':
        raise RuntimeError('Development artifact test status is not sealed.')

    selection_summary = read_json(SELECTION_SUMMARY)
    secondary_summary = read_json(SECONDARY_SUMMARY)
    if secondary_summary['test_data_status'] != 'header_only_not_loaded_or_evaluated':
        raise RuntimeError('Secondary analysis test status is not sealed.')
    forecast_weight = float(secondary_summary['internal_family_winners'][
        'secondary_cutoff_aligned_forecast_reranker']['parameters']['forecast_weight'])
    if forecast_weight != 0.0:
        raise RuntimeError('Secondary analysis selected nonzero forecast weight.')

    train, validation, combined = load_combined(module)
    catalog = module.build_catalog(train)
    if len(catalog) != 145:
        raise RuntimeError('Unexpected catalog size.')
    refit, checks = fit_components(module, combined, catalog, dev)

    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        'artifact_version': 'recommendation_pretest_v1',
        'freeze_status': 'recommendation_pretest_frozen',
        'freeze_name': PRETEST_TAG,
        'created_at_utc': created_at,
        'git_branch': state['branch'],
        'git_commit_before_freeze': state['head'],
        'design_freeze_commit': state['design_freeze_commit'],
        'primary_selection_commit': state['primary_selection_commit'],
        'secondary_analysis_commit': state['secondary_analysis_commit'],
        'refit_status': 'refit_on_train_plus_validation_for_one_time_sealed_test',
        'refit_source_splits': ['train', 'validation'],
        **refit,
        'selected_family': dev['selected_family'],
        'selected_model_name': dev['selected_model_name'],
        'selected_parameters': dev['selected_parameters'],
        'internal_family_winners': dev['internal_family_winners'],
        'hybrid_parameters': dev['hybrid_parameters'],
        'herd_parameters': dev['herd_parameters'],
        'scoring_contract': {
            'candidate_exclusion': 'exclude all train-plus-validation user history during test ranking',
            'tie_breaker': 'repo_id ascending',
            'top_k': 10,
            'forecast_weight': 0.0,
            'static_content_in_primary': False,
        },
        'test_data_status': 'header_only_not_loaded_or_evaluated',
    }

    PRETEST_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump(artifact, PRETEST_ARTIFACT)
    reload_difference = verify_reload(artifact, joblib.load(PRETEST_ARTIFACT))
    artifact_hash = sha256(PRETEST_ARTIFACT)

    input_paths = [TRAIN, VALID, TEST, DESIGN_MANIFEST, DESIGN_METADATA,
                   DEV_ARTIFACT, DEV_METADATA, SELECTION_SUMMARY,
                   SECONDARY_SUMMARY, TRAIN_SCRIPT, Path(__file__).resolve()]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    relative_test = str(TEST.relative_to(ROOT))
    if input_hashes[relative_test] != design_manifest['input_hashes'][relative_test]:
        raise RuntimeError('Test file hash changed after design freeze.')

    summary = {
        'status': 'recommendation_pretest_refit_complete',
        'artifact_version': 'recommendation_pretest_v1',
        'selected_family': artifact['selected_family'],
        'selected_model_name': artifact['selected_model_name'],
        'selected_parameters': artifact['selected_parameters'],
        'hybrid_parameters': artifact['hybrid_parameters'],
        'herd_parameters': artifact['herd_parameters'],
        'train_rows': int(len(train)),
        'validation_rows': int(len(validation)),
        'combined_rows': int(len(combined)),
        'combined_users': int(combined['user_id'].nunique()),
        'combined_repositories': int(combined['repo_id'].nunique()),
        'duplicate_user_repo_pairs': int(combined.duplicated(['user_id', 'repo_id']).sum()),
        'component_validation': checks,
        'artifact_reload_maximum_absolute_difference': reload_difference,
        'test_header_columns': state['test_header_columns'],
        'test_data_status': 'header_only_not_loaded_or_evaluated',
    }
    write_json(REFIT_SUMMARY, summary)

    metadata = {
        'freeze_status': 'recommendation_pretest_frozen',
        'freeze_name': PRETEST_TAG,
        'created_at_utc': created_at,
        'artifact_version': 'recommendation_pretest_v1',
        'git_branch': state['branch'],
        'git_commit_before_freeze': state['head'],
        'design_freeze_commit': state['design_freeze_commit'],
        'primary_selection_commit': state['primary_selection_commit'],
        'secondary_analysis_commit': state['secondary_analysis_commit'],
        'selected_family': artifact['selected_family'],
        'selected_model_name': artifact['selected_model_name'],
        'selected_parameters': artifact['selected_parameters'],
        'hybrid_parameters': artifact['hybrid_parameters'],
        'herd_parameters': artifact['herd_parameters'],
        'refit_data': {
            'train_rows': int(len(train)),
            'validation_rows': int(len(validation)),
            'combined_rows': int(len(combined)),
            'combined_users': int(combined['user_id'].nunique()),
            'combined_repositories': int(combined['repo_id'].nunique()),
            'minimum_user_history': int(combined.groupby('user_id').size().min()),
            'maximum_user_history': int(combined.groupby('user_id').size().max()),
            'minimum_item_count': float(refit['item_counts'].min()),
            'maximum_item_count': float(refit['item_counts'].max()),
        },
        'component_validation': checks,
        'artifact': {
            'path': str(PRETEST_ARTIFACT.relative_to(ROOT)),
            'sha256': artifact_hash,
            'reload_maximum_absolute_difference': reload_difference,
        },
        'input_hashes': input_hashes,
        'verified_design_hashes': verified_design_hashes,
        'selection_evidence': {
            'development_artifact_version': dev['artifact_version'],
            'development_artifact_sha256': dev_hash,
            'primary_validation_metrics': dev_meta['selected_external_validation_metrics'],
            'secondary_forecast_weight': forecast_weight,
            'static_content_in_primary': False,
            'selection_status': selection_summary.get('status', 'primary_selection_complete'),
        },
        'scoring_contract': artifact['scoring_contract'],
        'software': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__,
            'joblib': joblib.__version__,
        },
        'test_data_status': 'header_only_not_loaded_or_evaluated',
    }
    write_json(PRETEST_METADATA, metadata)

    manifest = {
        'freeze_name': PRETEST_TAG,
        'artifact_version': 'recommendation_pretest_v1',
        'git_commit_before_freeze': state['head'],
        'artifact_path': str(PRETEST_ARTIFACT.relative_to(ROOT)),
        'artifact_sha256': artifact_hash,
        'metadata_path': str(PRETEST_METADATA.relative_to(ROOT)),
        'metadata_sha256': sha256(PRETEST_METADATA),
        'refit_summary_path': str(REFIT_SUMMARY.relative_to(ROOT)),
        'refit_summary_sha256': sha256(REFIT_SUMMARY),
        'input_hashes': input_hashes,
        'test_data_status': 'header_only_not_loaded_or_evaluated',
    }
    write_json(MANIFEST, manifest)

    print('=' * 104)
    print('RECOMMENDATION PRETEST FREEZE')
    print('=' * 104)
    print('Artifact version:', artifact['artifact_version'])
    print('Selected model:', artifact['selected_model_name'])
    print('Combined rows:', len(combined))
    print('Combined users:', combined['user_id'].nunique())
    print('Combined repositories:', combined['repo_id'].nunique())
    print('Fit matrix shape:', checks['fit_matrix_shape'])
    print('Fit matrix nonzero:', checks['fit_matrix_nonzero'])
    print('Artifact path:', PRETEST_ARTIFACT.relative_to(ROOT))
    print('Artifact SHA256:', artifact_hash)
    print('Reload maximum difference:', reload_difference)
    print('Manifest path:', MANIFEST.relative_to(ROOT))
    print('Test status: header_only_not_loaded_or_evaluated')
    print('No recommendation test values were loaded or evaluated.')


if __name__ == '__main__':
    main()
