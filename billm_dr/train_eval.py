import os
import random
import uuid
import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score
from billm_dr.model import DReKGNN_PyTorch
import pandas as pd
from billm_dr.metrics import calculate_ranking_metrics


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_knn_graph(sim_matrix, k=7):
    """构建归一化 KNN 图，用于消除噪声边"""
    n = sim_matrix.shape[0]
    knn_graph = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = -1
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        knn_graph[i, top_k_idx] = 1.0
    row_sum = knn_graph.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return knn_graph / row_sum


class BPR_BCE_Loss(nn.Module):
    def __init__(self, device, pos_weight=20.0, margin=1.0, bpr_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        self.margin = margin
        self.bpr_weight = bpr_weight

    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        pos_logits = logits[labels == 1]
        neg_logits = logits[labels == 0]

        if len(pos_logits) > 0 and len(neg_logits) > 0:
            indices = torch.randint(0, len(neg_logits), (len(pos_logits),))
            neg_sample = neg_logits[indices]
            ranking_loss = torch.nn.functional.relu(self.margin - (pos_logits - neg_sample)).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=logits.device)
        return bce_loss + self.bpr_weight * ranking_loss


# =====================================================================
# 模式 1: 单次 8:1:1 划分 + UUID 防碰撞的 Top-3 SWA 融合 (适合快速验证)
# =====================================================================

def run_single_eval(config):
    print("\n" + "=" * 50)
    print("🚀 开始单次验证评估 (含 Top-3 SWA 防碰撞融合)")
    print("=" * 50)

    set_global_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_path = config['feat_dir']
    dataset_path = config['proc_dir'] + "dataset.pt"
    save_dir = config['proc_dir']

    run_id = str(uuid.uuid4())[:6]

    d_feat = torch.from_numpy(np.load(f"{feature_path}drug_features.npy")).float().to(device)
    s_feat = torch.from_numpy(np.load(f"{feature_path}disease_features.npy")).float().to(device)
    data = torch.load(dataset_path, weights_only=False)

    adj_dd = torch.from_numpy(data['adj_dd']).float().to(device)
    adj_ss = torch.from_numpy(data['adj_ss']).float().to(device)
    adj_ds = torch.from_numpy(data['adj_ds']).float().to(device)

    train_dataset = TensorDataset(torch.from_numpy(data['train_x']), torch.from_numpy(data['train_y']).float())
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = DReKGNN_PyTorch(
        input_dim=4096, hidden_dim=config['hidden_dim'],
        dropout=config['dropout'], edge_drop=config['edge_drop'],
        num_drugs=adj_dd.shape[0], num_diseases=adj_ss.shape[0]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    criterion = BPR_BCE_Loss(device=device, pos_weight=config['pos_weight'], margin=config['margin'],
                             bpr_weight=config['bpr_weight'])

    K = 3
    top_k_models = []
    best_aupr = 0.0
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_labels in train_loader:
            batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            agg_drug, agg_disease = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
            logits = model.predict(agg_drug, agg_disease, batch_x.long())
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                val_x = torch.from_numpy(data['val_x']).to(device).long()
                v_logits = model.predict(agg_d, agg_s, val_x)
                v_probs = torch.sigmoid(v_logits).cpu().numpy()
                v_aupr = average_precision_score(data['val_y'], v_probs)

                if v_aupr > best_aupr:
                    best_aupr = v_aupr
                    patience_counter = 0
                    print(
                        f"Epoch {epoch:03d} | Loss: {total_loss / len(train_loader):.4f} | Val AUPR: {v_aupr:.4f} 🌟 历史新高!")
                else:
                    patience_counter += 1
                    print(
                        f"Epoch {epoch:03d} | Loss: {total_loss / len(train_loader):.4f} | Val AUPR: {v_aupr:.4f} | Patience: {patience_counter}/{config['patience']}")

                if len(top_k_models) < K:
                    save_path = os.path.join(save_dir, f"swa_top{len(top_k_models) + 1}_{run_id}.pth")
                    torch.save(model.state_dict(), save_path)
                    top_k_models.append((v_aupr, epoch, save_path))
                    top_k_models.sort(key=lambda x: x[0], reverse=True)
                elif v_aupr > top_k_models[-1][0]:
                    save_path = top_k_models[-1][2]
                    torch.save(model.state_dict(), save_path)
                    top_k_models[-1] = (v_aupr, epoch, save_path)
                    top_k_models.sort(key=lambda x: x[0], reverse=True)

            if patience_counter >= config['patience']:
                print(f"\n🛑 触发早停！验证集在连续 {config['patience'] * 10} 轮内未创历史新高。")
                break

    # ====== 测试阶段 ======
    test_x = torch.from_numpy(data['test_x']).to(device).long()
    test_y = data['test_y']
    t_probs_total = np.zeros(len(test_y))

    print(f"\n🔄 融合 Epoch 概率: {', '.join([str(m[1]) for m in top_k_models])}")
    for idx, (aupr_val, ep, path) in enumerate(top_k_models):
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
            t_logits = model.predict(agg_d, agg_s, test_x)
            t_probs_total += torch.sigmoid(t_logits).cpu().numpy()

        # 🚀 修改点 1: 拦截清理逻辑，永久保存最佳模型权重
        if os.path.exists(path):
            if idx == 0:  # idx为0的是AUPR最高的第一名模型
                best_model_path = os.path.join(save_dir, "best_model.pth")
                # 覆盖旧的best_model.pth避免重命名报错
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                os.rename(path, best_model_path)
                print(f"💾 最佳模型参数 (Epoch {ep}) 已永久保存至:\n   {best_model_path}")
            else:
                os.remove(path)  # 清理第二名和第三名的临时权重

    final_t_probs = t_probs_total / len(top_k_models)
    preds_50 = (final_t_probs > 0.5).astype(int)

    print("\n" + "=" * 22 + " 单次评估最终融合结果汇报 " + "=" * 22)
    print(f"🚀 核心排序指标:")
    print(f"   AUC:       {roc_auc_score(test_y, final_t_probs):.4f}")
    print(f"   AUPR:      {average_precision_score(test_y, final_t_probs):.4f}")
    print(f"📊 标准分类指标:")
    print(f"   Accuracy:  {accuracy_score(test_y, preds_50):.4f}")
    print(f"   Precision: {precision_score(test_y, preds_50):.4f}")
    print(f"   Recall:    {recall_score(test_y, preds_50):.4f}")
    print(f"   F1-score:  {f1_score(test_y, preds_50):.4f}")
    print("=" * 70)

    # 🚀 修改点 2: 将最终的预测结果打包为 CSV 并保存在 processed 文件夹
    save_csv_path = os.path.join(save_dir, "single_eval_predictions.csv")
    df_preds = pd.DataFrame({
        'Drug_ID': data['test_x'][:, 0].astype(int),
        'Disease_ID': data['test_x'][:, 1].astype(int),
        'True_Label': test_y.astype(int),
        'Pred_Prob': final_t_probs
    })
    df_preds.to_csv(save_csv_path, index=False)
    print(f"\n💾 预测结果明细已保存至:\n   {save_csv_path}")


# =====================================================================
# 模式 2: 严谨 10 折交叉验证 (纯内存变量融合版)
# =====================================================================
import os
import uuid
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score
from billm_dr.metrics import calculate_ranking_metrics  # 确保你已经创建了 metrics.py
from billm_dr.model import DReKGNN_PyTorch  # 确保模型已导入


def run_10fold_eval(config):
    """
    运行 10 折交叉验证，包含纯内存 SWA 融合，全局排序指标计算，并将原始预测结果保存为 CSV。
    """
    print("\n" + "=" * 60)
    print("🚀 开始 10-Fold 交叉验证 (纯内存变量融合版 + 全局指标 + 预测落盘)")
    print("=" * 60)

    # 1. 设置全局种子并配置设备
    set_global_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载特征与图结构矩阵
    try:
        d_feat = torch.from_numpy(np.load(f"{config['feat_dir']}drug_features.npy")).float().to(device)
        s_feat = torch.from_numpy(np.load(f"{config['feat_dir']}disease_features.npy")).float().to(device)
    except FileNotFoundError:
        print("❌ 错误: 未找到特征文件，请确保特征提取已成功运行。")
        return

    try:
        import scipy.io as sio
        mat = sio.loadmat(config['mat_path'])
        drug_sim = mat['drug'].astype(np.float32)
        disease_sim = mat['disease'].astype(np.float32)
        adj = mat['didr']
    except Exception as e:
        print(f"❌ 错误: 加载 .mat 文件失败: {e}")
        return

    # 确保 adj 的 shape 为 (num_drugs, num_diseases)
    if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
        adj = adj.T

    print(f"📊 图维度: 药物={adj.shape[0]}, 疾病={adj.shape[1]}")

    # 预先计算同构图的 KNN 边 (消除弱连接噪声)
    knn_k = int(config.get('knn_k', 7))
    adj_dd = torch.from_numpy(get_knn_graph(drug_sim, k=knn_k)).float().to(device)
    adj_ss = torch.from_numpy(get_knn_graph(disease_sim, k=knn_k)).float().to(device)

    # 获取所有的正样本和负样本索引
    pos_idx = np.argwhere(adj == 1)
    neg_idx_all = np.argwhere(adj == 0)

    print(f"📈 样本统计: 正样本={len(pos_idx)}, 负样本={len(neg_idx_all)}")

    kf = KFold(n_splits=10, shuffle=True, random_state=config['seed'])

    # 动态获取 TOP_K 配置，默认 6000
    TOP_K = config.get('top_k', 6000)

    # 初始化统计字典
    all_metrics = {
        'AUC': [], 'AUPR': [], f'Recall@{TOP_K}': [], f'NDCG@{TOP_K}': [],
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []
    }

    # 用于保存所有预测结果的列表
    all_predictions = []

    # ================= 开始 10 折循环 =================
    for fold, (train_val_pos_idx, test_pos_idx) in enumerate(kf.split(pos_idx)):
        print(f"\n" + "-" * 50)
        print(f"🏁 正在进行第 {fold + 1}/10 折训练...")
        print("-" * 50)

        # 3. 动态切分本折的数据
        # 从 9 份数据中再切出 1 份作为验证集 (保证 train:val:test = 8:1:1)
        tr_pos_idx, val_pos_idx = train_test_split(
            train_val_pos_idx, test_size=1 / 9.0, random_state=config['seed'] + fold
        )

        train_pos = pos_idx[tr_pos_idx]
        val_pos = pos_idx[val_pos_idx]
        test_pos = pos_idx[test_pos_idx]

        # 负样本切分策略：保持真实的测试/验证分布
        neg_train_val_idx, neg_test_idx = train_test_split(
            np.arange(len(neg_idx_all)), test_size=0.1, random_state=config['seed'] + fold
        )
        neg_train_idx, neg_val_idx = train_test_split(
            neg_train_val_idx, test_size=1 / 9.0, random_state=config['seed'] + fold
        )

        train_neg_all = neg_idx_all[neg_train_idx]
        val_neg = neg_idx_all[neg_val_idx]
        test_neg = neg_idx_all[neg_test_idx]

        # 训练集负采样：控制正负比例
        NEG_RATIO = int(config.get('neg_ratio', config.get('pos_weight', 10)))
        sample_size = min(len(train_pos) * NEG_RATIO, len(train_neg_all))
        train_neg = train_neg_all[np.random.choice(len(train_neg_all), sample_size, replace=False)]

        # 组装完整的训练、验证、测试集
        train_x = np.vstack([train_pos, train_neg])
        train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])

        val_x = np.vstack([val_pos, val_neg])
        val_y = np.hstack([np.ones(len(val_pos)), np.zeros(len(val_neg))])

        test_x = np.vstack([test_pos, test_neg])
        test_y = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])

        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).float())
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        # 4. 防止数据穿越：仅使用训练集正样本构建 adj_ds
        train_adj_ds = np.zeros_like(adj, dtype=np.float32)
        for d, s in train_pos:
            train_adj_ds[d, s] = 1.0
        row_sum = train_adj_ds.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        adj_ds = torch.from_numpy(train_adj_ds / row_sum).float().to(device)

        # 5. 初始化模型、优化器与损失函数
        model = DReKGNN_PyTorch(
            input_dim=config.get('input_dim', 4096),  # 兼容可能动态探测的特征维度
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            edge_drop=config['edge_drop'],
            num_drugs=adj.shape[0],
            num_diseases=adj.shape[1]
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        criterion = BPR_BCE_Loss(
            device=device,
            pos_weight=config['pos_weight'],
            margin=config['margin'],
            bpr_weight=config['bpr_weight']
        )

        # 6. SWA Top-K 内存追踪器初始化
        K = 3
        top_k_models = []
        best_aupr = 0.0
        patience_counter = 0

        # ================= 训练循环 =================
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            total_loss = 0
            for batch_x, batch_labels in train_loader:
                batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                agg_drug, agg_disease = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                logits = model.predict(agg_drug, agg_disease, batch_x.long())
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # 每 10 轮在验证集上评估一次
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                    val_x_tensor = torch.from_numpy(val_x).to(device).long()
                    v_logits = model.predict(agg_d, agg_s, val_x_tensor)
                    v_probs = torch.sigmoid(v_logits).cpu().numpy()
                    v_aupr = average_precision_score(val_y, v_probs)

                    # 早停机制判断
                    if v_aupr > best_aupr:
                        best_aupr = v_aupr
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # 克隆参数到内存，脱离计算图，节省显存
                    state_dict_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    # 动态更新 Top-K 候选池
                    if len(top_k_models) < K:
                        top_k_models.append((v_aupr, epoch, state_dict_copy))
                        top_k_models.sort(key=lambda x: x[0], reverse=True)
                    elif v_aupr > top_k_models[-1][0]:
                        top_k_models[-1] = (v_aupr, epoch, state_dict_copy)
                        top_k_models.sort(key=lambda x: x[0], reverse=True)

                if patience_counter >= config['patience']:
                    print(f"🛑 [第{fold + 1}折] 触发早停! 验证集最高 AUPR: {best_aupr:.4f} (Epoch {epoch})")
                    break

        # ================= 测试集评估与结果收集 =================
        test_x_tensor = torch.from_numpy(test_x).to(device).long()
        t_probs_total = np.zeros(len(test_y))

        print(f"🔄 融合本折表现最佳的 Epoch 权重: {[m[1] for m in top_k_models]}")
        for idx, (aupr_val, ep, mem_state_dict) in enumerate(top_k_models):
            model.load_state_dict(mem_state_dict)
            model.eval()
            with torch.no_grad():
                agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                t_logits = model.predict(agg_d, agg_s, test_x_tensor)
                t_probs_total += torch.sigmoid(t_logits).cpu().numpy()

        # 对 K 个模型的概率取平均
        final_t_probs = t_probs_total / len(top_k_models)
        preds_50 = (final_t_probs > 0.5).astype(int)

        # 收集本次折叠中每一个预测的明细，落盘用于后续复用/画图
        for i in range(len(test_x)):
            all_predictions.append({
                'Fold': fold + 1,
                'Drug_ID': int(test_x[i][0]),
                'Disease_ID': int(test_x[i][1]),
                'True_Label': int(test_y[i]),
                'Pred_Prob': float(final_t_probs[i])
            })

        # 计算标准指标
        fold_auc = roc_auc_score(test_y, final_t_probs)
        fold_aupr = average_precision_score(test_y, final_t_probs)
        fold_acc = accuracy_score(test_y, preds_50)
        fold_prec = precision_score(test_y, preds_50)
        fold_rec = recall_score(test_y, preds_50)
        fold_f1 = f1_score(test_y, preds_50)

        # 🚀 计算全局排序指标 (Global Ranking)
        # 提取前 TOP_K 个预测概率最高的索引
        top_k_indices = np.argsort(final_t_probs)[::-1][:TOP_K]

        # 计算全局 Recall@K
        hits = np.sum(test_y[top_k_indices])
        total_positives = np.sum(test_y)
        global_recall_at_k = hits / total_positives if total_positives > 0 else 0.0

        # 计算全局 NDCG@K
        try:
            from sklearn.metrics import ndcg_score
            global_ndcg_at_k = ndcg_score([test_y], [final_t_probs], k=TOP_K)
        except Exception as e:
            print(f"⚠️ 计算 NDCG 出错: {e}，将设为 0.0")
            global_ndcg_at_k = 0.0

        print(f"✅ 第 {fold + 1} 折成绩 -> AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f} | "
              f"Global Recall@{TOP_K}: {global_recall_at_k:.4f} | Global NDCG@{TOP_K}: {global_ndcg_at_k:.4f}")

        # 录入本折分数供最后计算全局均值和方差
        all_metrics['AUC'].append(fold_auc)
        all_metrics['AUPR'].append(fold_aupr)
        all_metrics[f'Recall@{TOP_K}'].append(global_recall_at_k)
        all_metrics[f'NDCG@{TOP_K}'].append(global_ndcg_at_k)
        all_metrics['Accuracy'].append(fold_acc)
        all_metrics['Precision'].append(fold_prec)
        all_metrics['Recall'].append(fold_rec)
        all_metrics['F1'].append(fold_f1)

    # ================= 10 折结束：数据落盘与终极汇报 =================

    # 保存预测结果为 CSV
    os.makedirs(config['proc_dir'], exist_ok=True)
    save_path = os.path.join(config['proc_dir'], "10fold_raw_predictions.csv")
    df_predictions = pd.DataFrame(all_predictions)
    try:
        df_predictions.to_csv(save_path, index=False)
        print(f"\n💾 所有原始预测数据已保存至:\n   {save_path}")
    except Exception as e:
        print(f"\n⚠️ 保存 CSV 失败: {e}")

    print("\n" + "=" * 55)
    print("🏆 10-Fold 交叉验证最终总成绩汇报")
    print("=" * 55)
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"   {metric_name:<15}: {mean_val:.4f} ± {std_val:.4f}")
    print("=" * 55)