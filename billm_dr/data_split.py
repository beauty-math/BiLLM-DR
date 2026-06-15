import numpy as np
import scipy.io as sio
import torch
import os
from sklearn.model_selection import train_test_split

def get_knn_graph(sim_matrix, k=7):
    """构建归一化 KNN 图"""
    n = sim_matrix.shape[0]
    knn_graph = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = -1
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        knn_graph[i, top_k_idx] = 1.0

    row_sum = knn_graph.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return knn_graph / row_sum

def run_data_split(mat_path, output_dir, k_neighbors=7, seed=42):
    print(f"\n--- 开始数据集划分与图构建 (SOTA 论文同款严谨评测标准, Seed={seed}) ---")
    mat = sio.loadmat(mat_path)

    # 提取相似度矩阵，获取当前数据集真实的药物和疾病数量
    drug_sim = mat['drug'].astype(np.float32)
    disease_sim = mat['disease'].astype(np.float32)

    n_drugs = drug_sim.shape[0]
    n_diseases = disease_sim.shape[0]

    # 动态转置，自动适应 Fdataset / Cdataset
    adj = mat['didr']
    if adj.shape == (n_diseases, n_drugs):
        adj = adj.T

    # 构建归一化的 KNN 邻接矩阵
    adj_dd = get_knn_graph(drug_sim, k=k_neighbors)
    adj_ss = get_knn_graph(disease_sim, k=k_neighbors)

    pos_idx = np.argwhere(adj == 1)
    neg_idx_all = np.argwhere(adj == 0)

    # 1. 🚀 对【正样本】进行 8:1:1 划分
    train_pos, vt_pos = train_test_split(pos_idx, test_size=0.2, random_state=seed)
    val_pos, test_pos = train_test_split(vt_pos, test_size=0.5, random_state=seed)

    # 2. 🚀 对【全量负样本】进行 8:1:1 划分 (这保证了测试集极度不平衡，符合顶刊标准)
    train_neg_all, vt_neg = train_test_split(neg_idx_all, test_size=0.2, random_state=seed)
    val_neg, test_neg = train_test_split(vt_neg, test_size=0.5, random_state=seed)

    # 3. 🚀 训练集处理：将原本的 1:1 采样，改为 1:10 的非对称采样
    np.random.seed(seed)
    NEG_RATIO = 20  # 让负样本是正样本的 10 倍

    # 确保不会超出负样本总数
    sample_size = min(len(train_pos) * NEG_RATIO, len(train_neg_all))
    train_neg_indices = np.random.choice(len(train_neg_all), sample_size, replace=False)
    train_neg = train_neg_all[train_neg_indices]

    # 4. 分别组装特征与标签
    train_x = np.vstack([train_pos, train_neg])
    train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])

    # 验证集和测试集保留全量真实分布 (以 Fdataset 为例，正负比例约 1:95)
    val_x = np.vstack([val_pos, val_neg])
    val_y = np.hstack([np.ones(len(val_pos)), np.zeros(len(val_neg))])

    test_x = np.vstack([test_pos, test_neg])
    test_y = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    # 5. 构建严谨的训练级图谱 (绝对不包含验证集和测试集的关联边)
    train_adj_ds = np.zeros_like(adj, dtype=np.float32)
    for d, s in train_pos:
        train_adj_ds[d, s] = 1.0

    row_sum = train_adj_ds.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    train_adj_ds_norm = train_adj_ds / row_sum

    data = {
        'train_x': train_x, 'train_y': train_y,
        'val_x': val_x, 'val_y': val_y,
        'test_x': test_x, 'test_y': test_y,
        'adj_dd': adj_dd, 'adj_ss': adj_ss, 'adj_ds': train_adj_ds_norm
    }
    torch.save(data, os.path.join(output_dir, "dataset.pt"))
    print(f"✅ 数据划分完成！")
    print(f"   -> 训练集: {len(train_pos)} 正样本, {len(train_neg)} 负样本 (比例 1:1)")
    print(f"   -> 验证集: {len(val_pos)} 正样本, {len(val_neg)} 负样本 (全量盲测)")
    print(f"   -> 测试集: {len(test_pos)} 正样本, {len(test_neg)} 负样本 (全量盲测)")