import os
import glob
import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd


def parse_ks(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def latest_raw_prediction(project_dir, dataset):
    pattern = os.path.join(project_dir, 'results', dataset, f'{dataset}_10fold_*_raw_predictions.csv')
    files = [p for p in glob.glob(pattern) if '/ablations/' not in p]
    if not files:
        raise FileNotFoundError(f'No raw prediction file found: {pattern}')
    return max(files, key=os.path.getmtime)


def ndcg_binary(labels_sorted, k, total_pos):
    k = min(int(k), len(labels_sorted))
    if k <= 0 or total_pos <= 0:
        return 0.0
    rel = np.asarray(labels_sorted[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_k = min(k, int(total_pos))
    if ideal_k <= 0:
        return 0.0
    idcg = float(np.sum(discounts[:ideal_k]))
    return dcg / idcg if idcg > 0 else 0.0


def compute_topk(raw_path, ks):
    df = pd.read_csv(raw_path)
    required = {'Fold', 'True_Label', 'Pred_Prob'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{raw_path} missing columns: {sorted(missing)}')

    rows = []
    for fold, part in df.groupby('Fold', sort=True):
        part = part.sort_values('Pred_Prob', ascending=False).reset_index(drop=True)
        y = part['True_Label'].to_numpy(dtype=int)
        total_pos = int(y.sum())
        n = len(y)
        for k0 in ks:
            k = min(int(k0), n)
            top_y = y[:k]
            hits = int(top_y.sum())
            rows.append({
                'Fold': int(fold),
                'K': int(k0),
                'Effective_K': int(k),
                'Total_Pos': total_pos,
                'Hits@K': hits,
                'Precision@K': hits / k if k else 0.0,
                'Recall@K': hits / total_pos if total_pos else 0.0,
                'NDCG@K': ndcg_binary(y, k, total_pos),
            })
    per_fold = pd.DataFrame(rows)

    summary_rows = []
    for k, part in per_fold.groupby('K', sort=True):
        row = {'K': int(k), 'Folds': int(part['Fold'].nunique())}
        for metric in ['Hits@K', 'Precision@K', 'Recall@K', 'NDCG@K']:
            values = part[metric].astype(float).to_numpy()
            row[f'{metric}_mean'] = float(np.mean(values))
            row[f'{metric}_std'] = float(np.std(values))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return per_fold, summary


def main():
    parser = argparse.ArgumentParser(description='Compute Top-K recommendation metrics from saved 10-fold raw predictions.')
    parser.add_argument('--project-dir', default='/home/zqgaopengfei/project/zym')
    parser.add_argument('--datasets', default=os.getenv('DATASETS', os.getenv('DATASET_NAME', 'Fdataset,Cdataset')))
    parser.add_argument('--ks', default=os.getenv('TOPK_LIST', '10,20,50,6000'))
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]
    ks = parse_ks(args.ks)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for dataset in datasets:
        raw_path = latest_raw_prediction(args.project_dir, dataset)
        out_dir = os.path.join(args.project_dir, 'results', dataset, 'topk')
        os.makedirs(out_dir, exist_ok=True)
        per_fold, summary = compute_topk(raw_path, ks)
        stem = os.path.basename(raw_path).replace('_raw_predictions.csv', '')
        per_fold_path = os.path.join(out_dir, f'{stem}_topk_{timestamp}_per_fold.csv')
        summary_path = os.path.join(out_dir, f'{stem}_topk_{timestamp}_summary.csv')
        per_fold.to_csv(per_fold_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f'[{dataset}] raw: {raw_path}')
        print(f'[{dataset}] per-fold saved: {per_fold_path}')
        print(f'[{dataset}] summary saved: {summary_path}')
        print(summary.to_string(index=False))
        print('')


if __name__ == '__main__':
    main()
