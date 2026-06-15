
import os
import re
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from billm_dr.model import DReKGNN_PyTorch

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '/home/zqgaopengfei/project/zym')


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def short_desc(text: str, max_len: int = 90) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip()[:max_len]


def unit(x, eps=1e-9):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_bundle(dataset, device):
    root = os.path.join(PROJECT_ROOT, 'data', dataset)
    proc = os.path.join(root, 'processed')
    feat = os.path.join(root, 'features')
    data = torch.load(os.path.join(proc, 'dataset.pt'), map_location='cpu', weights_only=False)
    d_feat = torch.from_numpy(np.load(os.path.join(feat, 'drug_features.npy'))).float().to(device)
    s_feat = torch.from_numpy(np.load(os.path.join(feat, 'disease_features.npy'))).float().to(device)
    adj_dd = torch.from_numpy(data['adj_dd']).float().to(device)
    adj_ss = torch.from_numpy(data['adj_ss']).float().to(device)
    adj_ds = torch.from_numpy(data['adj_ds']).float().to(device)
    model = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1], hidden_dim=512, dropout=0.5, edge_drop=0.25,
        num_drugs=d_feat.shape[0], num_diseases=s_feat.shape[0]
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(proc, 'best_model.pth'), map_location=device))
    model.eval()
    with torch.no_grad():
        drug_emb, disease_emb = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
    drug_df = pd.read_csv(os.path.join(root, 'drug_desc.csv'))
    disease_df = pd.read_csv(os.path.join(root, 'disease_desc.csv'))
    return {
        'dataset': dataset,
        'root': root,
        'out_dir': os.path.join(PROJECT_ROOT, 'results', dataset, 'visualization_query_space'),
        'data': data,
        'model': model,
        'device': device,
        'drug_emb_t': drug_emb,
        'disease_emb_t': disease_emb,
        'drug_emb': drug_emb.detach().cpu().numpy(),
        'disease_emb': disease_emb.detach().cpu().numpy(),
        'drug_ids': drug_df['db_id'].astype(str).tolist(),
        'drug_desc': drug_df['Description'].astype(str).tolist(),
        'disease_ids': disease_df['omim_id'].astype(str).tolist(),
        'disease_desc': disease_df['Description'].astype(str).tolist(),
    }


def predict_probs(bundle, disease_idx):
    n = bundle['drug_emb_t'].shape[0]
    pairs = torch.tensor([[i, disease_idx] for i in range(n)], dtype=torch.long, device=bundle['device'])
    with torch.no_grad():
        logits = bundle['model'].predict(bundle['drug_emb_t'], bundle['disease_emb_t'], pairs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def query_vector(bundle, disease_idx):
    with torch.no_grad():
        q = bundle['model'].bilinear_w(bundle['disease_emb_t'][disease_idx:disease_idx + 1])
    return q.detach().cpu().numpy()[0]


def make_target_query_plot(bundle, target_omim, top_k=10, known_k=10, bg_k=80):
    dataset = bundle['dataset']
    os.makedirs(bundle['out_dir'], exist_ok=True)
    if target_omim not in bundle['disease_ids']:
        print(f'[{dataset}] target {target_omim} not found')
        return
    disease_idx = bundle['disease_ids'].index(target_omim)
    probs = predict_probs(bundle, disease_idx)
    q = query_vector(bundle, disease_idx)
    drug = bundle['drug_emb']

    # Query-space: disease is transformed by W and compared to drug embeddings.
    x_all = unit(np.vstack([q[None, :], drug]))
    q_u = x_all[0:1]
    drug_u = x_all[1:]
    cos = cosine_similarity(drug_u, q_u).reshape(-1)

    adj = bundle['data']['adj_ds']
    known = np.where(adj[:, disease_idx] > 0)[0]
    unknown = np.where(adj[:, disease_idx] == 0)[0]
    top_pred = unknown[np.argsort(probs[unknown])[::-1][:top_k]]
    known_top = known[np.argsort(probs[known])[::-1][:min(known_k, len(known))]] if len(known) else np.array([], dtype=int)

    selected = np.unique(np.concatenate([top_pred, known_top]))
    bg_pool = np.setdiff1d(np.argsort(cos)[::-1][:max(bg_k, top_k + known_k + 20)], selected)
    bg = bg_pool[:bg_k]
    plot_drugs = np.concatenate([selected, bg])

    mat = np.vstack([q_u, drug_u[plot_drugs]])
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(mat)
    xy = xy - xy[0]  # target-centered plot; target query is exactly at origin.
    target_xy = xy[0]
    drug_xy = xy[1:]

    role_map = {int(i): 'Background nearest drugs' for i in bg}
    for i in known_top:
        role_map[int(i)] = 'Known associated drug'
    for i in top_pred:
        role_map[int(i)] = 'Top predicted candidate'
    # If a known drug is also in top predicted, show it as known to avoid double legend.
    for i in set(top_pred).intersection(set(known_top)):
        role_map[int(i)] = 'Known associated drug'

    rows = [{
        'Dataset': dataset, 'Target_OMIM': target_omim, 'Role': 'Target disease query',
        'Drug_Index': '', 'Drug_ID': target_omim, 'Pred_Prob': 1.0, 'Cosine_to_Query': 1.0,
        'X': float(target_xy[0]), 'Y': float(target_xy[1]),
        'Description_Short': short_desc(bundle['disease_desc'][disease_idx], 140),
    }]
    for j, i in enumerate(plot_drugs):
        rows.append({
            'Dataset': dataset, 'Target_OMIM': target_omim, 'Role': role_map[int(i)],
            'Drug_Index': int(i), 'Drug_ID': bundle['drug_ids'][int(i)],
            'Pred_Prob': float(probs[int(i)]), 'Cosine_to_Query': float(cos[int(i)]),
            'X': float(drug_xy[j, 0]), 'Y': float(drug_xy[j, 1]),
            'Description_Short': short_desc(bundle['drug_desc'][int(i)], 140),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(bundle['out_dir'], f'{dataset}_target_{target_omim}_query_space_points.csv')
    df.to_csv(csv_path, index=False)

    colors = {
        'Background nearest drugs': '#cbd5e1',
        'Known associated drug': '#16a34a',
        'Top predicted candidate': '#2563eb',
        'Target disease query': '#dc2626',
    }
    markers = {
        'Background nearest drugs': 'o',
        'Known associated drug': 'o',
        'Top predicted candidate': '^',
        'Target disease query': '*',
    }
    sizes = {
        'Background nearest drugs': 18,
        'Known associated drug': 90,
        'Top predicted candidate': 105,
        'Target disease query': 260,
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=240)
    for role in ['Background nearest drugs', 'Known associated drug', 'Top predicted candidate', 'Target disease query']:
        part = df[df['Role'] == role]
        if part.empty:
            continue
        ax.scatter(part['X'], part['Y'], s=sizes[role], c=colors[role], marker=markers[role],
                   alpha=0.28 if role == 'Background nearest drugs' else 0.94,
                   edgecolors='white' if role != 'Background nearest drugs' else 'none',
                   linewidths=0.8, label=role)
    # Label target and top five predicted plus top five known.
    label_df = pd.concat([
        df[df['Role'] == 'Target disease query'],
        df[df['Role'] == 'Top predicted candidate'].sort_values('Pred_Prob', ascending=False).head(5),
        df[df['Role'] == 'Known associated drug'].sort_values('Pred_Prob', ascending=False).head(5),
    ]).drop_duplicates(subset=['Role', 'Drug_ID'])
    for _, r in label_df.iterrows():
        ax.text(r['X'], r['Y'], str(r['Drug_ID']), fontsize=7, ha='left', va='bottom')
    ax.axhline(0, color='#e5e7eb', lw=0.8, zorder=0)
    ax.axvline(0, color='#e5e7eb', lw=0.8, zorder=0)
    ax.set_title(f'{dataset}: target-centered query space OMIM {target_omim}', fontsize=12)
    ax.set_xlabel('Query-space PC1 centered at target')
    ax.set_ylabel('Query-space PC2 centered at target')
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    png_path = os.path.join(bundle['out_dir'], f'{dataset}_target_{target_omim}_query_space_pca.png')
    fig.savefig(png_path)
    plt.close(fig)

    # Rank-radius plot: faithful to prediction score, target is intentionally at center.
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=240)
    focus = df[df['Role'].isin(['Known associated drug', 'Top predicted candidate'])].copy()
    focus = focus.sort_values('Pred_Prob', ascending=False).head(top_k + known_k)
    if not focus.empty:
        score = focus['Pred_Prob'].to_numpy()
        score_norm = (score - score.min()) / (score.max() - score.min() + 1e-9)
        radii = 0.18 + (1.0 - score_norm) * 0.78
        angles = np.linspace(0, 2*np.pi, len(focus), endpoint=False)
        focus['RX'] = radii * np.cos(angles)
        focus['RY'] = radii * np.sin(angles)
        for role in ['Known associated drug', 'Top predicted candidate']:
            part = focus[focus['Role'] == role]
            ax.scatter(part['RX'], part['RY'], s=sizes[role], c=colors[role], marker=markers[role],
                       edgecolors='white', linewidths=0.8, label=role)
        for _, r in focus.head(10).iterrows():
            ax.text(r['RX'], r['RY'], str(r['Drug_ID']), fontsize=7, ha='left', va='bottom')
    ax.scatter([0], [0], s=280, c=colors['Target disease query'], marker='*', edgecolors='white', linewidths=0.8, label='Target disease query')
    ax.text(0, 0, target_omim, fontsize=8, ha='left', va='bottom')
    for radius in [0.25, 0.5, 0.75, 1.0]:
        circle = plt.Circle((0, 0), radius, color='#e5e7eb', fill=False, lw=0.7, zorder=0)
        ax.add_artist(circle)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{dataset}: prediction-score radial view OMIM {target_omim}', fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    radial_path = os.path.join(bundle['out_dir'], f'{dataset}_target_{target_omim}_score_radial.png')
    fig.savefig(radial_path)
    plt.close(fig)

    print(f'[{dataset}] {target_omim}: saved {png_path}')
    print(f'[{dataset}] {target_omim}: saved {radial_path}')
    print(f'[{dataset}] {target_omim}: saved {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default=os.getenv('DATASETS', 'Fdataset,Cdataset'))
    parser.add_argument('--targets', default=os.getenv('TARGET_DISEASES', '168600,600807,157300'))
    parser.add_argument('--top-k', type=int, default=int(os.getenv('TOP_K', '10')))
    parser.add_argument('--known-k', type=int, default=int(os.getenv('KNOWN_K', '10')))
    parser.add_argument('--device', default=os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    print(f'Query-space visualization | datasets={args.datasets} targets={args.targets} device={device}', flush=True)
    for dataset in parse_list(args.datasets):
        bundle = load_bundle(dataset, device)
        for target in parse_list(args.targets):
            make_target_query_plot(bundle, target, top_k=args.top_k, known_k=args.known_k)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
