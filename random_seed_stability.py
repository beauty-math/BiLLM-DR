import os
from dataclasses import asdict
from typing import List

import pandas as pd
import torch

from parameter_sensitivity import (
    BASE_PARAMS,
    SensitivityConfig,
    load_features,
    load_mat_data,
    train_single_run,
)


def parse_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(',') if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item) for item in parse_list(text)]


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1',
        'Best_Val_AUPR', 'Train_Positives', 'Train_Negatives',
        'Val_Positives', 'Test_Positives',
    ]
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors='coerce').dropna()
        rows.append({
            'Metric': metric,
            'Mean': values.mean(),
            'Std': values.std(ddof=0),
            'Min': values.min(),
            'Max': values.max(),
        })
    return pd.DataFrame(rows)


def run_dataset(project_root: str, dataset: str, seeds: List[int], timestamp: str):
    result_dir = os.path.join(project_root, 'results', dataset, 'seed_stability')
    os.makedirs(result_dir, exist_ok=True)

    cfg = SensitivityConfig()
    cfg.project_root = project_root
    cfg.dataset_name = dataset
    cfg.save_predictions = False

    params = dict(BASE_PARAMS)
    run_tag = (
        f'{dataset}_seed_stability_neg{params["neg_ratio"]}'
        f'_posw{params["pos_weight"]}_bpr{params["bpr_weight"]}'
        f'_knn{params["knn_k"]}_{timestamp}'
    )

    print('=' * 80, flush=True)
    print(f'Dataset: {dataset}', flush=True)
    print(f'Result dir: {result_dir}', flush=True)
    print(f'Seeds: {seeds}', flush=True)
    print(f'Params: {params}', flush=True)
    print(f'Epochs: {cfg.epochs}, patience: {cfg.patience}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    d_feat, s_feat = load_features(cfg, device)
    drug_sim, disease_sim, adj = load_mat_data(cfg)

    rows = []
    for idx, seed in enumerate(seeds, start=1):
        cfg.seed = int(seed)
        print(f'\n[{idx:02d}/{len(seeds)}] seed={seed}', flush=True)
        metrics, _ = train_single_run(cfg, params, d_feat, s_feat, drug_sim, disease_sim, adj)
        row = {
            'Dataset': dataset,
            'Seed': seed,
            **params,
            **metrics,
        }
        rows.append(row)
        print(
            f"  AUC={metrics['AUC']:.4f} AUPR={metrics['AUPR']:.4f} "
            f"F1={metrics['F1']:.4f} BestValAUPR={metrics['Best_Val_AUPR']:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(
            os.path.join(result_dir, f'{run_tag}_per_seed_partial.csv'),
            index=False,
        )

    per_seed = pd.DataFrame(rows)
    summary = metric_summary(per_seed)

    per_seed_path = os.path.join(result_dir, f'{run_tag}_per_seed.csv')
    summary_path = os.path.join(result_dir, f'{run_tag}_summary.csv')
    config_path = os.path.join(result_dir, f'{run_tag}_config.txt')
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f'dataset: {dataset}\n')
        f.write(f'seeds: {seeds}\n')
        f.write(f'params: {params}\n')
        f.write(f'epochs: {cfg.epochs}\n')
        f.write(f'patience: {cfg.patience}\n')
        f.write(f'config: {asdict(cfg)}\n')

    print('\nSaved:', flush=True)
    print(f'  per-seed: {per_seed_path}', flush=True)
    print(f'  summary:  {summary_path}', flush=True)
    print(f'  config:   {config_path}', flush=True)
    print('\nSummary:', flush=True)
    print(summary.to_string(index=False), flush=True)
    return per_seed_path, summary_path


def main():
    project_root = os.environ.get('PROJECT_ROOT', '/home/zqgaopengfei/project/zym')
    datasets = parse_list(os.environ.get('DATASETS', os.environ.get('DATASET_NAME', 'Fdataset,Cdataset')))
    seeds = parse_int_list(os.environ.get('SEEDS', '1,7,42,2024,3407'))
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    print('Random seed stability experiment', flush=True)
    print(f'Project root: {project_root}', flush=True)
    print(f'Datasets: {datasets}', flush=True)
    print(f'Seeds: {seeds}', flush=True)

    all_rows = []
    for dataset in datasets:
        per_seed_path, _ = run_dataset(project_root, dataset, seeds, timestamp)
        all_rows.append(pd.read_csv(per_seed_path))

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        overall_dir = os.path.join(project_root, 'results', 'seed_stability')
        os.makedirs(overall_dir, exist_ok=True)
        all_path = os.path.join(overall_dir, f'all_datasets_seed_stability_{timestamp}_per_seed.csv')
        all_df.to_csv(all_path, index=False)
        print(f'\nAll dataset per-seed table saved to: {all_path}', flush=True)


if __name__ == '__main__':
    main()
