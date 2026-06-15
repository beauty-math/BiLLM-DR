import torch
import torch.nn as nn
import torch.nn.functional as F

class DReKGNN_PyTorch(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, dropout=0.6, edge_drop=0.2, num_drugs=663, num_diseases=409):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_drop = nn.Dropout(dropout)
        self.edge_drop = edge_drop

        self.drug_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.disease_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.intra_w_drug_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.intra_w_disease_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inter_w_drug_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inter_w_disease_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.intra_w_drug_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.intra_w_disease_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inter_w_drug_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inter_w_disease_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.bilinear_w = nn.Linear(hidden_dim, hidden_dim)

        self.drug_bias = nn.Embedding(num_drugs, 1)
        self.disease_bias = nn.Embedding(num_diseases, 1)
        nn.init.zeros_(self.drug_bias.weight)
        nn.init.zeros_(self.disease_bias.weight)

        # 🚀 核心新增：自适应多跳的可学习权重 (0-hop, 1-hop, 2-hop)
        # 初始化为 [1.0, 1.0, 1.0]，让模型在训练中自己寻找最佳融合比例
        self.hop_weights_drug = nn.Parameter(torch.ones(3))
        self.hop_weights_disease = nn.Parameter(torch.ones(3))

    def forward(self, d_feat_llm, s_feat_llm, adj_dd, adj_ss, adj_ds):
        d_feat = self.feat_drop(F.normalize(d_feat_llm, p=2, dim=1))
        s_feat = self.feat_drop(F.normalize(s_feat_llm, p=2, dim=1))
        h_d_0 = self.drug_proj(d_feat)
        h_s_0 = self.disease_proj(s_feat)

        adj_dd_do = F.dropout(adj_dd, p=self.edge_drop, training=self.training)
        adj_ss_do = F.dropout(adj_ss, p=self.edge_drop, training=self.training)
        adj_ds_do = F.dropout(adj_ds, p=self.edge_drop, training=self.training)

        # 🚀 核心修复：补全所有图在 DropEdge 之后的动态度归一化 (Degree Normalization)
        # 异构图 (疾病 -> 药物)
        adj_sd_do = adj_ds_do.t()
        col_sum_sd = adj_sd_do.sum(dim=1, keepdim=True)
        col_sum_sd[col_sum_sd == 0] = 1.0
        adj_sd_norm = adj_sd_do / col_sum_sd

        # 同构图 (药物 -> 药物)
        col_sum_dd = adj_dd_do.sum(dim=1, keepdim=True)
        col_sum_dd[col_sum_dd == 0] = 1.0
        adj_dd_norm = adj_dd_do / col_sum_dd

        # 同构图 (疾病 -> 疾病)
        col_sum_ss = adj_ss_do.sum(dim=1, keepdim=True)
        col_sum_ss[col_sum_ss == 0] = 1.0
        adj_ss_norm = adj_ss_do / col_sum_ss

        # --- 第一跳 (全部换成 norm 后的矩阵) ---
        intra_d_1 = self.intra_w_drug_1(torch.matmul(adj_dd_norm, h_d_0))
        intra_s_1 = self.intra_w_disease_1(torch.matmul(adj_ss_norm, h_s_0))
        inter_d_1 = self.inter_w_drug_1(torch.matmul(adj_ds_do, h_s_0))
        inter_s_1 = self.inter_w_disease_1(torch.matmul(adj_sd_norm, h_d_0))

        h_d_1 = intra_d_1 + inter_d_1 + h_d_0
        h_s_1 = intra_s_1 + inter_s_1 + h_s_0

        # --- 第二跳 ---
        intra_d_2 = self.intra_w_drug_2(torch.matmul(adj_dd_norm, h_d_1))
        intra_s_2 = self.intra_w_disease_2(torch.matmul(adj_ss_norm, h_s_1))
        inter_d_2 = self.inter_w_drug_2(torch.matmul(adj_ds_do, h_s_1))
        inter_s_2 = self.inter_w_disease_2(torch.matmul(adj_sd_norm, h_d_1))

        h_d_2 = intra_d_2 + inter_d_2 + h_d_1
        h_s_2 = intra_s_2 + inter_s_2 + h_s_1

        # 🚀 核心新增：应用自适应多跳权重
        # 使用 Softmax 保证权重之和为 1，且全部为正
        w_d = F.softmax(self.hop_weights_drug, dim=0)
        w_s = F.softmax(self.hop_weights_disease, dim=0)

        # 按照学到的比例进行完美混合
        agg_drug = w_d[0] * h_d_0 + w_d[1] * h_d_1 + w_d[2] * h_d_2
        agg_disease = w_s[0] * h_s_0 + w_s[1] * h_s_1 + w_s[2] * h_s_2

        return F.elu(agg_drug), F.elu(agg_disease)

    def predict(self, agg_drug, agg_disease, batch_indices):
        d_idx = batch_indices[:, 0]
        s_idx = batch_indices[:, 1]
        batch_d = agg_drug[d_idx]
        batch_s = agg_disease[s_idx]
        score = (batch_d * self.bilinear_w(batch_s)).sum(dim=-1)
        score = score + self.drug_bias(d_idx).squeeze() + self.disease_bias(s_idx).squeeze()
        return score

