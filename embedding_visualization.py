
import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from billm_dr.model import DReKGNN_PyTorch


PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '/home/zqgaopengfei/project/zym')


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def short_desc(text: str, max_len: int = 70) -> str:
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text[:max_len]


def load_embeddings(dataset: str, device: torch.device):
    root = os.path.join(PROJECT_ROOT, 'data', dataset)
    proc = os.path.join(root, 'processed')
    feat = os.path.join(root, 'features')
    result_dir = os.path.join(PROJECT_ROOT, 'results', dataset, 'visualization')
    os.makedirs(result_dir, exist_ok=True)

    data = torch.load(os.path.join(proc, 'dataset.pt'), map_location='cpu', weights_only=False)
    d_feat = torch.from_numpy(np.load(os.path.join(feat, 'drug_features.npy'))).float().to(device)
    s_feat = torch.from_numpy(np.load(os.path.join(feat, 'disease_features.npy'))).float().to(device)
    adj_dd = torch.from_numpy(data['adj_dd']).float().to(device)
    adj_ss = torch.from_numpy(data['adj_ss']).float().to(device)
    adj_ds = torch.from_numpy(data['adj_ds']).float().to(device)

    model = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1],
        hidden_dim=512,
        dropout=0.5,
        edge_drop=0.25,
        num_drugs=d_feat.shape[0],
        num_diseases=s_feat.shape[0],
    ).to(device)
    state = torch.load(os.path.join(proc, 'best_model.pth'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        drug_emb, disease_emb = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
    return {
        'root': root,
        'proc': proc,
        'result_dir': result_dir,
        'data': data,
        'model': model,
        'drug_emb': drug_emb.detach().cpu().numpy(),
        'disease_emb': disease_emb.detach().cpu().numpy(),
        'drug_emb_t': drug_emb,
        'disease_emb_t': disease_emb,
        'device': device,
    }


def load_metadata(dataset: str):
    root = os.path.join(PROJECT_ROOT, 'data', dataset)
    drug_df = pd.read_csv(os.path.join(root, 'drug_desc.csv'))
    disease_df = pd.read_csv(os.path.join(root, 'disease_desc.csv'))
    drug_ids = drug_df['db_id'].astype(str).tolist() if 'db_id' in drug_df.columns else [f'Drug_{i}' for i in range(len(drug_df))]
    disease_ids = disease_df['omim_id'].astype(str).tolist() if 'omim_id' in disease_df.columns else [f'Disease_{i}' for i in range(len(disease_df))]
    drug_desc = drug_df['Description'].astype(str).tolist() if 'Description' in drug_df.columns else drug_ids
    disease_desc = disease_df['Description'].astype(str).tolist() if 'Description' in disease_df.columns else disease_ids
    return drug_ids, disease_ids, drug_desc, disease_desc


def compute_coordinates(drug_emb, disease_emb, seed=42):
    x = np.vstack([drug_emb, disease_emb])
    node_type = np.array(['Drug'] * len(drug_emb) + ['Disease'] * len(disease_emb))
    pca = PCA(n_components=2, random_state=seed)
    pca_xy = pca.fit_transform(x)
    perplexity = min(30, max(5, (len(x) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=seed)
    tsne_xy = tsne.fit_transform(x)
    return pca_xy, tsne_xy, node_type, pca.explained_variance_ratio_


def save_global_plots(dataset, out_dir, coord_df, pca_var):
    colors = {'Drug': '#3b82f6', 'Disease': '#ef4444'}
    for method, xcol, ycol, title_extra in [
        ('pca', 'PCA1', 'PCA2', f'PCA explained var={pca_var[0]:.2%},{pca_var[1]:.2%}'),
        ('tsne', 'TSNE1', 'TSNE2', 't-SNE'),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=220)
        for typ in ['Drug', 'Disease']:
            part = coord_df[coord_df['Node_Type'] == typ]
            ax.scatter(part[xcol], part[ycol], s=10 if typ == 'Drug' else 14,
                       c=colors[typ], alpha=0.62, label=typ, linewidths=0)
        ax.set_title(f'{dataset} learned embeddings ({title_extra})', fontsize=12)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend(frameon=False, loc='best')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{dataset}_global_{method}.png'))
        plt.close(fig)


def predict_for_disease(bundle, disease_idx: int):
    model = bundle['model']
    device = bundle['device']
    n_drugs = bundle['drug_emb_t'].shape[0]
    pairs = torch.tensor([[i, disease_idx] for i in range(n_drugs)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.predict(bundle['drug_emb_t'], bundle['disease_emb_t'], pairs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def save_target_plots(dataset, out_dir, coord_df, bundle, drug_ids, disease_ids, drug_desc, disease_desc, target_omim: str, top_k: int):
    if target_omim not in disease_ids:
        print(f'[{dataset}] target OMIM {target_omim} not found, skip.')
        return None
    disease_idx = disease_ids.index(target_omim)
    probs = predict_for_disease(bundle, disease_idx)
    adj = bundle['data']['adj_ds']
    known = np.where(adj[:, disease_idx] > 0)[0]
    unknown = np.where(adj[:, disease_idx] == 0)[0]
    top_unknown = unknown[np.argsort(probs[unknown])[::-1][:top_k]]
    known_top = known[np.argsort(probs[known])[::-1][:min(top_k, len(known))]] if len(known) else np.array([], dtype=int)

    rows = []
    for rank, i in enumerate(top_unknown, start=1):
        rows.append({
            'Dataset': dataset,
            'Target_OMIM': target_omim,
            'Target_Disease_Index': disease_idx,
            'Rank': rank,
            'Drug_Index': int(i),
            'Drug_ID': drug_ids[int(i)],
            'Pred_Prob': float(probs[int(i)]),
            'Known_Association': int(adj[int(i), disease_idx] > 0),
            'Drug_Description_Short': short_desc(drug_desc[int(i)], 120),
            'Disease_Description_Short': short_desc(disease_desc[disease_idx], 160),
        })
    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(out_dir, f'{dataset}_target_{target_omim}_top{top_k}_predictions.csv')
    pred_df.to_csv(pred_path, index=False)

    selected = []
    selected.append(('Disease', disease_idx, 'Target disease', target_omim, 1.0))
    for i in known_top:
        selected.append(('Drug', int(i), 'Known associated drug', drug_ids[int(i)], probs[int(i)]))
    for i in top_unknown:
        selected.append(('Drug', int(i), 'Top predicted candidate', drug_ids[int(i)], probs[int(i)]))

    sel_rows = []
    for typ, idx, role, label, score in selected:
        row = coord_df[(coord_df.Node_Type == typ) & (coord_df.Node_Index == idx)].iloc[0].to_dict()
        row.update({'Role': role, 'Plot_Label': label, 'Pred_Prob': float(score)})
        sel_rows.append(row)
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(out_dir, f'{dataset}_target_{target_omim}_plot_points.csv'), index=False)

    marker = {'Target disease': '*', 'Known associated drug': 'o', 'Top predicted candidate': '^'}
    color = {'Target disease': '#dc2626', 'Known associated drug': '#16a34a', 'Top predicted candidate': '#2563eb'}
    for method, xcol, ycol in [('pca','PCA1','PCA2'), ('tsne','TSNE1','TSNE2')]:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=220)
        bg = coord_df.sample(min(len(coord_df), 1000), random_state=42)
        ax.scatter(bg[xcol], bg[ycol], s=8, c='#cbd5e1', alpha=0.25, linewidths=0, label='All nodes')
        for role in ['Known associated drug', 'Top predicted candidate', 'Target disease']:
            part = sel_df[sel_df['Role'] == role]
            if part.empty:
                continue
            size = 80 if role != 'Target disease' else 220
            ax.scatter(part[xcol], part[ycol], s=size, c=color[role], marker=marker[role],
                       edgecolors='white', linewidths=0.8, alpha=0.95, label=role)
            # Label target and top five candidates only, keeping the plot readable.
            label_part = part if role == 'Target disease' else part.head(5)
            for _, r in label_part.iterrows():
                ax.text(r[xcol], r[ycol], str(r['Plot_Label']), fontsize=7, ha='left', va='bottom')
        ax.set_title(f'{dataset}: target disease OMIM {target_omim} ({method.upper()})', fontsize=12)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{dataset}_target_{target_omim}_{method}.png'))
        plt.close(fig)
    return pred_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default=os.getenv('DATASETS', 'Fdataset,Cdataset'))
    parser.add_argument('--targets', default=os.getenv('TARGET_DISEASES', '168600,600807,157300'))
    parser.add_argument('--top-k', type=int, default=int(os.getenv('TOP_K', '10')))
    parser.add_argument('--device', default=os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    targets = parse_list(args.targets)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    print(f'Embedding visualization analysis | datasets={datasets} targets={targets} device={device}')

    for dataset in datasets:
        print('\n' + '='*72)
        print(f'Dataset: {dataset}')
        bundle = load_embeddings(dataset, device)
        drug_ids, disease_ids, drug_desc, disease_desc = load_metadata(dataset)
        pca_xy, tsne_xy, node_type, pca_var = compute_coordinates(bundle['drug_emb'], bundle['disease_emb'])
        n_drugs = len(bundle['drug_emb'])
        node_ids = drug_ids + disease_ids
        descriptions = [short_desc(x, 120) for x in drug_desc] + [short_desc(x, 120) for x in disease_desc]
        coord_df = pd.DataFrame({
            'Dataset': dataset,
            'Node_Type': node_type,
            'Node_Index': list(range(n_drugs)) + list(range(len(disease_ids))),
            'Node_ID': node_ids,
            'Description_Short': descriptions,
            'PCA1': pca_xy[:,0], 'PCA2': pca_xy[:,1],
            'TSNE1': tsne_xy[:,0], 'TSNE2': tsne_xy[:,1],
        })
        out_dir = bundle['result_dir']
        coord_path = os.path.join(out_dir, f'{dataset}_embedding_coordinates.csv')
        coord_df.to_csv(coord_path, index=False)
        save_global_plots(dataset, out_dir, coord_df, pca_var)
        print(f'Coordinates saved: {coord_path}')
        print(f'Global plots saved under: {out_dir}')
        for target in targets:
            path = save_target_plots(dataset, out_dir, coord_df, bundle, drug_ids, disease_ids, drug_desc, disease_desc, target, args.top_k)
            if path:
                print(f'Target {target} top predictions: {path}')

    print('\nVisualization analysis completed.')


if __name__ == '__main__':
    main()
