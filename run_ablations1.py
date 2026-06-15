import os
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
from billm_dr.model import DReKGNN_PyTorch
from billm_dr.train_eval import BPR_BCE_Loss, set_global_seed, get_knn_graph


# =========================================================
# ⚙️ 实验配置区 (Configuration) - 您可以自由修改这里的一切
# =========================================================
class AblationConfig:
    # --- 1. 数据集设置 ---
    ROOT_DIR = os.environ.get("DATA_ROOT", "/home/zqgaopengfei/project/zym/data")
    DATASET_NAME = "Cdataset"  # 👈 随意切换: "Fdataset", "Cdataset", "Bdataset" 等

    # 自动推导路径 (基于传统 .mat 格式)
    # 如果您的新数据集纯靠 CSV，请参考之前的指导自行替换此处加载逻辑
    MAT_PATH = os.path.join(ROOT_DIR, DATASET_NAME, f"{DATASET_NAME}.mat")
    FEAT_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "features/")
    PROC_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "processed/")

    # 保存结果的目录
    OUTPUT_DIR = os.path.join(PROC_DIR, "ablation_results/")

    # --- 2. 验证策略 ---
    NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))  # 👈 消融实验用几折交叉验证？(为了快可以填 3 或 5，发论文用 10)
    EPOCHS = int(os.environ.get("EPOCHS", 1000))  # 👈 消融实验跑多少轮？(为了快可以填 100 或 200)
    SEED = int(os.environ.get("SEED", 42))

    # --- 3. 模型超参数 ---
    BATCH_SIZE = 512
    HIDDEN_DIM = 512
    DROPOUT = float(os.environ.get("DROPOUT", 0.7))
    EDGE_DROP = float(os.environ.get("EDGE_DROP", 0.5))
    LR = float(os.environ.get("LR", 1e-4))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-3))
    POS_WEIGHT = float(os.environ.get("POS_WEIGHT", 10.0))  # 负采样比例 (1:10)
    MARGIN = 1.0  # BPR 损失的 Margin


# 实例化配置
CFG = AblationConfig()
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# =========================================================
# 🔬 消融实验方案定义 (Ablation Conditions)
# =========================================================
# 您可以在这里自由添加、修改或删除要测试的实验组合
ABLATIONS = [
    {
        "name": "Ours_Full",
        "use_llm_feats": True,
        "bpr_weight": 0.5,
        "fix_hop_weights": False
    },
    {
        "name": "w/o_Text_Feat",
        "use_llm_feats": False,
        "bpr_weight": 0.5,
        "fix_hop_weights": False
    },
    {
        "name": "w/o_BPR_Loss",
        "use_llm_feats": True,
        "bpr_weight": 0.0,
        "fix_hop_weights": False
    },
    {
        "name": "w/o_MultiHop_Weights",
        "use_llm_feats": True,
        "bpr_weight": 0.5,
        "fix_hop_weights": True
    }
]


# =========================================================
# 🚀 核心运行逻辑
# =========================================================
def run_ablation_experiment(ablation_config):
    print(f"\n{'=' * 60}")
    print(f"🔬 正在运行消融实验: {ablation_config['name']}")
    print(f"{'=' * 60}")

    set_global_seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载数据与构建图结构 ---
    print(f"📦 加载数据集: {CFG.DATASET_NAME}...")
    try:
        mat = sio.loadmat(CFG.MAT_PATH)
        drug_sim = mat['drug'].astype(np.float32)
        disease_sim = mat['disease'].astype(np.float32)
        adj = mat['didr']
    except Exception as e:
        print(f"❌ 读取 .mat 文件失败: {e}\n如果您使用的是纯 CSV 数据集，请修改此处代码使用 pd.read_csv。")
        return None

    # 动态转置
    if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
        adj = adj.T

    num_drugs, num_diseases = adj.shape

    # 构建 KNN 图
    adj_dd = torch.from_numpy(get_knn_graph(drug_sim, k=7)).float().to(device)
    adj_ss = torch.from_numpy(get_knn_graph(disease_sim, k=7)).float().to(device)

    # --- 2. 处理特征 (消融点 1) ---
    feat_dim = CFG.HIDDEN_DIM  # 默认维度
    if ablation_config['use_llm_feats']:
        try:
            d_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}drug_features.npy")).float().to(device)
            s_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}disease_features.npy")).float().to(device)
            feat_dim = d_feat.shape[1]
            print(f"   ✅ 启用大模型文本特征 (Dim: {feat_dim})")
        except FileNotFoundError:
            print(f"❌ 未找到特征文件: {CFG.FEAT_DIR}drug_features.npy。请先运行大模型提取脚本。")
            return None
    else:
        # ABLATION: 随机初始化特征
        d_feat = torch.randn(num_drugs, feat_dim).float().to(device)
        s_feat = torch.randn(num_diseases, feat_dim).float().to(device)
        print(f"   ⚠️ 禁用大模型特征，使用随机初始化向量 (Dim: {feat_dim})")

    # --- 3. 数据划分与 K 折验证 ---
    pos_idx = np.argwhere(adj == 1)
    neg_idx_all = np.argwhere(adj == 0)

    # 动态设定折数
    kf = KFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    metrics = {'AUC': [], 'AUPR': []}

    # 为了使输出整洁，不再打印每一折的详细训练日志
    for fold, (train_val_pos_idx, test_pos_idx) in enumerate(kf.split(pos_idx)):
        print(f"\n   ⏳ 正在训练 Fold {fold + 1}/{CFG.NUM_FOLDS} ...")

        # --- 切分数据 ---
        tr_pos_idx, val_pos_idx = train_test_split(train_val_pos_idx, test_size=0.1, random_state=CFG.SEED)
        train_pos = pos_idx[tr_pos_idx]
        val_pos = pos_idx[val_pos_idx]
        test_pos = pos_idx[test_pos_idx]

        neg_train_val_idx, neg_test_idx = train_test_split(np.arange(len(neg_idx_all)), test_size=0.1,
                                                           random_state=CFG.SEED)
        neg_train_idx, neg_val_idx = train_test_split(neg_train_val_idx, test_size=0.1, random_state=CFG.SEED)

        train_neg_all = neg_idx_all[neg_train_idx]
        val_neg = neg_idx_all[neg_val_idx]
        test_neg = neg_idx_all[neg_test_idx]

        # 负采样
        NEG_RATIO = int(CFG.POS_WEIGHT)
        sample_size = min(len(train_pos) * NEG_RATIO, len(train_neg_all))
        train_neg = train_neg_all[np.random.choice(len(train_neg_all), sample_size, replace=False)]

        train_x = np.vstack([train_pos, train_neg])
        train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        test_x = np.vstack([test_pos, test_neg])
        test_y = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])

        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).float())
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)

        # 构建防穿越的交互图
        train_adj_ds = np.zeros_like(adj, dtype=np.float32)
        for d, s in train_pos:
            train_adj_ds[d, s] = 1.0
        row_sum = train_adj_ds.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        adj_ds = torch.from_numpy(train_adj_ds / row_sum).float().to(device)

        # --- 4. 初始化模型 ---
        model = DReKGNN_PyTorch(
            input_dim=feat_dim,
            hidden_dim=CFG.HIDDEN_DIM,
            dropout=CFG.DROPOUT,
            edge_drop=CFG.EDGE_DROP,
            num_drugs=num_drugs,
            num_diseases=num_diseases
        ).to(device)

        # ABLATION Target 3: 固定多跳聚合权重
        if ablation_config['fix_hop_weights']:
            # 将多跳权重设为全0 (Softmax 后就是均等的 1/3)
            model.hop_weights_drug.data = torch.zeros(3).to(device)
            model.hop_weights_disease.data = torch.zeros(3).to(device)
            model.hop_weights_drug.requires_grad = False
            model.hop_weights_disease.requires_grad = False
            if fold == 0:  # 只打印一次
                print("   ⚠️ 已禁用自适应多跳权重，降级为简单均值聚合。")

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.LR,
                                      weight_decay=CFG.WEIGHT_DECAY)

        # ABLATION Target 2: 调整 BPR 损失权重
        criterion = BPR_BCE_Loss(
            device=device,
            pos_weight=CFG.POS_WEIGHT,
            margin=CFG.MARGIN,
            bpr_weight=ablation_config['bpr_weight']
        )
        if ablation_config['bpr_weight'] == 0 and fold == 0:
            print("   ⚠️ 已禁用 BPR 排序损失，仅使用 BCE 进行优化。")

        # --- 5. 训练循环 ---
        for epoch in range(1, CFG.EPOCHS + 1):
            model.train()
            for batch_x, batch_labels in train_loader:
                batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                agg_drug, agg_disease = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                logits = model.predict(agg_drug, agg_disease, batch_x.long())
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

        # --- 6. 评估 (盲测集) ---
        model.eval()
        test_x_tensor = torch.from_numpy(test_x).to(device).long()
        with torch.no_grad():
            agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
            t_logits = model.predict(agg_d, agg_s, test_x_tensor)
            t_probs = torch.sigmoid(t_logits).cpu().numpy()

        fold_auc = roc_auc_score(test_y, t_probs)
        fold_aupr = average_precision_score(test_y, t_probs)
        metrics['AUC'].append(fold_auc)
        metrics['AUPR'].append(fold_aupr)
        print(f"      -> 该折测试结果: AUC={fold_auc:.4f} | AUPR={fold_aupr:.4f}")

    mean_auc = np.mean(metrics['AUC'])
    mean_aupr = np.mean(metrics['AUPR'])
    print(f"\n✨ {ablation_config['name']} 实验完成! 最终平均 AUC: {mean_auc:.4f} | AUPR: {mean_aupr:.4f}")

    return {"Model (Ablation)": ablation_config['name'], "Mean AUROC": mean_auc, "Mean AUPR": mean_aupr}


def main():
    results = []
    # 逐一运行消融实验
    for ab_config in ABLATIONS:
        res = run_ablation_experiment(ab_config)
        if res:
            results.append(res)

    # 汇总并保存结果
    if results:
        df = pd.DataFrame(results)

        # 生成带有数据集名称和折数的动态文件名
        csv_filename = f"ablation_{CFG.DATASET_NAME}_{CFG.NUM_FOLDS}fold.csv"
        save_path = os.path.join(CFG.OUTPUT_DIR, csv_filename)

        df.to_csv(save_path, index=False)

        print("\n" + "=" * 50)
        print("🏆 消融实验 (Ablation Study) 汇总结果")
        print("=" * 50)
        print(f"📍 数据集: {CFG.DATASET_NAME} | 验证折数: {CFG.NUM_FOLDS}-Fold")
        print("-" * 50)
        print(df.to_markdown(index=False))
        print("=" * 50)
        print(f"\n💾 汇总数据已成功保存至:\n   {save_path}")


if __name__ == "__main__":
    main()