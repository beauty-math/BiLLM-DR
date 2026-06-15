import numpy as np
import pandas as pd
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


def load_csv_matrix(filepath):
    """自适应读取 CSV 矩阵文件"""
    try:
        df = pd.read_csv(filepath, header=None)
        if df.iloc[0].apply(lambda x: isinstance(x, str)).any():
            df = pd.read_csv(filepath, index_col=0)
        return df.values.astype(np.float32)
    except Exception as e:
        raise ValueError(f"读取 CSV 文件 {filepath} 失败: {e}")


def run_data_split(config, k_neighbors=7):
    seed = config['seed']
    output_dir = config['proc_dir']
    print(f"\n--- 开始数据集划分与图构建 (CSV 版本, SOTA 黄金比例, Seed={seed}) ---")

    # 从 config 中动态获取 csv 路径
    drug_sim = load_csv_matrix(config['drug_sim_csv'])
    disease_sim = load_csv_matrix(config['dis_sim_csv'])
    adj = load_csv_matrix(config['drug_dis_csv'])

    n_drugs = drug_sim.shape[0]
    n_diseases = disease_sim.shape[0]

    # 动态转置适配
    if adj.shape == (n_diseases, n_drugs):
        adj = adj.T

    # 构建归一化的 KNN 邻接矩阵
    adj_dd = get_knn_graph(drug_sim, k=k_neighbors)
    adj_ss = get_knn_graph(disease_sim, k=k_neighbors)

    pos_idx = np.argwhere(adj == 1)
    neg_idx_all = np.argwhere(adj == 0)

    # 1. 对【正样本】进行 8:1:1 划分
    train_pos, vt_pos = train_test_split(pos_idx, test_size=0.2, random_state=seed)
    val_pos, test_pos = train_test_split(vt_pos, test_size=0.5, random_state=seed)

    # 2. 对【全量负样本】进行 8:1:1 划分
    train_neg_all, vt_neg = train_test_split(neg_idx_all, test_size=0.2, random_state=seed)
    val_neg, test_neg = train_test_split(vt_neg, test_size=0.5, random_state=seed)

    # 3. 训练集负采样：控制正负比例
    np.random.seed(seed)
    NEG_RATIO = int(config.get('neg_ratio', config.get('pos_weight', 10)))
    sample_size = min(len(train_pos) * NEG_RATIO, len(train_neg_all))
    train_neg_indices = np.random.choice(len(train_neg_all), sample_size, replace=False)
    train_neg = train_neg_all[train_neg_indices]

    # 4. 组装特征与标签
    train_x = np.vstack([train_pos, train_neg])
    train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])

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

    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, os.path.join(output_dir, "dataset.pt"))

    print(f"✅ 数据划分完成！")
    print(f"   -> 训练集: {len(train_pos)} 正样本, {len(train_neg)} 负样本")
    print(f"   -> 验证集: {len(val_pos)} 正样本, {len(val_neg)} 负样本")
    print(f"   -> 测试集: {len(test_pos)} 正样本, {len(test_neg)} 负样本")