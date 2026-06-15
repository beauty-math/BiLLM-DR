# import os
# import torch
# import numpy as np
# import pandas as pd
# import scipy.io as sio
#
# # 导入您的模型架构
# from model import DReKGNN_PyTorch
#
#
# # =========================================================
# # ⚙️ 案例分析配置区
# # =========================================================
# class CaseStudyConfig:
#     ROOT_DIR = "/root/autodl-tmp/zym/data/"
#     DATASET_NAME = "Fdataset"  # 案例分析最常用的数据集
#
#     MAT_PATH = os.path.join(ROOT_DIR, DATASET_NAME, f"{DATASET_NAME}.mat")
#     FEAT_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "features/")
#     PROC_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "processed/")
#
#     # 🚨 必须修改：请填入您训练好的最佳模型权重路径！
#     MODEL_WEIGHT_PATH = os.path.join(PROC_DIR, "best_model.pth")  # <--- 修改这里
#
#     DRUG_CSV = os.path.join(ROOT_DIR, DATASET_NAME, "drug_desc.csv")
#     DISEASE_CSV = os.path.join(ROOT_DIR, DATASET_NAME, "disease_desc.csv")
#
#     # 模型架构参数（必须与训练时保持绝对一致）
#     HIDDEN_DIM = 512
#     DROPOUT = 0.7
#
#     # 🎯 您想要进行案例分析的疾病关键词或 OMIM ID
#     # 104300 是阿尔茨海默症(AD)，168600 是帕金森病(PD)
#     TARGET_DISEASES = ["Alzheimer", "Parkinson", "104300", "168600"]
#     TOP_K = 10
#
#
# CFG = CaseStudyConfig()
#
#
# # =========================================================
# # 🛠️ 辅助函数：加载真实名称映射
# # =========================================================
# def get_name_mapping(csv_path, id_col_candidates, name_col_candidates):
#     """尝试从 CSV 中提取 ID 到 真实名称 的映射字典"""
#     df = pd.read_csv(csv_path)
#
#     # 寻找 ID 列
#     id_col = next((col for col in id_col_candidates if col in df.columns), df.columns[0])
#     # 寻找 Name 列 (如果没有 Name 列，就用 ID 代替)
#     name_col = next((col for col in name_col_candidates if col in df.columns), id_col)
#
#     mapping = dict(zip(df[id_col].astype(str).str.strip(), df[name_col].astype(str).str.strip()))
#     return mapping
#
#
# # =========================================================
# # 🚀 核心运行逻辑
# # =========================================================
# def main():
#     print(f"\n{'=' * 60}")
#     print(f"🔬 启动临床新药发现预测 (Case Study) - {CFG.DATASET_NAME}")
#     print(f"{'=' * 60}")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. 检查权重文件
#     if not os.path.exists(CFG.MODEL_WEIGHT_PATH):
#         print(f"❌ 找不到模型权重文件: {CFG.MODEL_WEIGHT_PATH}")
#         print("💡 请先运行训练脚本，保存一个 .pth 权重文件，然后修改 CFG.MODEL_WEIGHT_PATH。")
#         return
#
#     # 2. 读取原始 .mat 文件获取图结构和索引顺序
#     print("📦 正在加载图结构与文本信息...")
#     mat = sio.loadmat(CFG.MAT_PATH)
#     drug_sim = mat['drug'].astype(np.float32)
#     disease_sim = mat['disease'].astype(np.float32)
#     adj = mat['didr']
#
#     if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
#         adj = adj.T  # 确保 shape 为 (num_drugs, num_diseases)
#
#     num_drugs, num_diseases = adj.shape
#
#     # 提取按顺序排列的 ID
#     ordered_drug_ids = [str(i[0][0]).strip() for i in mat['Wrname']]
#     ordered_disease_ids = [str(i[0][0]).strip() for i in mat['Wdname']]
#
#     # 3. 读取 CSV 获取可读的名称
#     # 修改后的代码：
#     drug_map = get_name_mapping(CFG.DRUG_CSV, ['db_id', 'drug_id'],
#                                 ['Name', 'name', 'Description', 'description', 'db_id'])
#     disease_map = get_name_mapping(CFG.DISEASE_CSV, ['omim_id', 'disease_id'],
#                                    ['Name', 'name', 'Description', 'description', 'omim_id'])
#
#     # 4. 加载 GNN 消息传递所需的张量 (从 dataset.pt 加载以保证与训练时图结构完全一致)
#     dataset_path = os.path.join(CFG.PROC_DIR, "dataset.pt")
#     if not os.path.exists(dataset_path):
#         print("❌ 找不到 dataset.pt，请先运行数据切分。")
#         return
#
#     data = torch.load(dataset_path, weights_only=False)
#     adj_dd = torch.from_numpy(data['adj_dd']).float().to(device)
#     adj_ss = torch.from_numpy(data['adj_ss']).float().to(device)
#     adj_ds = torch.from_numpy(data['adj_ds']).float().to(device)
#
#     # 5. 加载 LLM 特征
#     d_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}drug_features.npy")).float().to(device)
#     s_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}disease_features.npy")).float().to(device)
#
#     # 6. 初始化并加载模型
#     print(f"📥 加载预训练模型权重...")
#     model = DReKGNN_PyTorch(
#         input_dim=d_feat.shape[1], hidden_dim=CFG.HIDDEN_DIM,
#         dropout=CFG.DROPOUT, edge_drop=0.0,  # 预测时不丢弃边
#         num_drugs=num_drugs, num_diseases=num_diseases
#     ).to(device)
#
#     model.load_state_dict(torch.load(CFG.MODEL_WEIGHT_PATH, map_location=device))
#     model.eval()
#
#     # 预先进行一次整图聚合
#     with torch.no_grad():
#         agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
#
#     # =========================================================
#     # 🔎 开始对每种目标疾病进行探索
#     # =========================================================
#     all_results = []
#
#     for kw in CFG.TARGET_DISEASES:
#         # 寻找匹配的疾病索引
#         target_dis_idx = -1
#         target_dis_id = ""
#         target_dis_name = ""
#
#         kw_lower = kw.lower()
#         for idx, d_id in enumerate(ordered_disease_ids):
#             name = disease_map.get(d_id, "Unknown")
#             if kw_lower in name.lower() or kw_lower in d_id.lower():
#                 target_dis_idx = idx
#                 target_dis_id = d_id
#                 target_dis_name = name
#                 break
#
#         if target_dis_idx == -1:
#             print(f"⚠️ 未能在数据集中找到与 '{kw}' 相关的疾病，已跳过。")
#             continue
#
#         print(f"\n" + "=" * 50)
#         print(f"🎯 正在分析疾病: {target_dis_name} (ID: {target_dis_id})")
#         print("=" * 50)
#
#         # 找出该疾病所有【未知的药物】 (即 adj 中为 0 的药物)
#         unknown_drug_indices = np.where(adj[:, target_dis_idx] == 0)[0]
#         print(f"   -> 发现 {len(unknown_drug_indices)} 种尚未证明与该疾病相关的候选药物。")
#
#         # 构建测试批次
#         test_x = torch.tensor([[d_idx, target_dis_idx] for d_idx in unknown_drug_indices]).long().to(device)
#
#         # 进行预测
#         with torch.no_grad():
#             t_logits = model.predict(agg_d, agg_s, test_x)
#             t_probs = torch.sigmoid(t_logits).cpu().numpy()
#
#         # 对概率进行降序排序，获取 Top-K
#         top_k_relative_idx = np.argsort(t_probs)[::-1][:CFG.TOP_K]
#
#         print(f"\n🏆 Top-{CFG.TOP_K} 潜在治疗药物推荐榜单:")
#         print(f"{'Rank':<5} | {'Drug ID':<12} | {'Probability':<12} | {'Drug Name/Desc'}")
#         print("-" * 80)
#
#         for rank, rel_idx in enumerate(top_k_relative_idx):
#             drug_idx = unknown_drug_indices[rel_idx]  # 真实的药物索引
#             prob = t_probs[rel_idx]
#
#             drug_id = ordered_drug_ids[drug_idx]
#             drug_name = drug_map.get(drug_id, "Unknown")
#
#             # 截断过长的名字以保持排版整洁
#             display_name = (drug_name[:45] + '...') if len(drug_name) > 45 else drug_name
#
#             print(f"{rank + 1:<5} | {drug_id:<12} | {prob:.6f}     | {display_name}")
#
#             # 保存到总结果中
#             all_results.append({
#                 "Disease": target_dis_name,
#                 "Disease_ID": target_dis_id,
#                 "Rank": rank + 1,
#                 "Drug_ID": drug_id,
#                 "Drug_Name": drug_name,
#                 "Score": prob
#             })
#
#     # 将结果保存为 CSV
#     if all_results:
#         df_results = pd.DataFrame(all_results)
#         csv_path = os.path.join(CFG.PROC_DIR, "case_study_top10.csv")
#         df_results.to_csv(csv_path, index=False)
#         print(f"\n💾 案例分析结果已成功保存至: {csv_path}")
#         print(
#             "💡 您可以将这些推荐出的药物名称拿到 CTD (Comparative Toxicogenomics Database) 或 ClinicalTrials 网站上去搜索验证，如果能找到证据，就可以直接写进论文的 Case Study 表格里啦！")
#
#
# if __name__ == "__main__":
#     main()


import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

# 导入您的模型架构
from billm_dr.model import DReKGNN_PyTorch


# =========================================================
# ⚙️ 案例分析配置区
# =========================================================
class CaseStudyConfig:
    ROOT_DIR = "/root/autodl-tmp/zym/data/"
    DATASET_NAME = "Fdataset"  # 案例分析最常用的数据集

    MAT_PATH = os.path.join(ROOT_DIR, DATASET_NAME, f"{DATASET_NAME}.mat")
    FEAT_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "features/")
    PROC_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "processed/")

    # 🚨 必须修改：请填入您训练好的最佳模型权重路径！
    MODEL_WEIGHT_PATH = os.path.join(PROC_DIR, "best_model.pth")

    DRUG_CSV = os.path.join(ROOT_DIR, DATASET_NAME, "drug_desc.csv")
    DISEASE_CSV = os.path.join(ROOT_DIR, DATASET_NAME, "disease_desc.csv")

    # 模型架构参数（必须与训练时保持绝对一致）
    HIDDEN_DIM = 512
    DROPOUT = 0.7

    # 🎯 您想要进行案例分析的疾病关键词或 OMIM ID
    TARGET_DISEASES = ["Alzheimer", "Parkinson", "104300", "168600"]
    TOP_K = 10

    # 🎲 用于背景噪音的随机药物 ID (对照组)
    RANDOM_DRUGS = ["DB00115", "DB01050", "DB00945", "DB01211", "DB00331", "DB00526", "DB01104"]


CFG = CaseStudyConfig()

# 💡 专业药名字典：防止图表标签出现长句或乱码
PROFESSIONAL_NAME_MAP = {
    "DB00843": "Donepezil", "DB01043": "Memantine", "DB00674": "Galantamine",
    "DB00382": "Tacrine", "DB00989": "Rivastigmine", "DB00810": "Biperiden",
    "DB00376": "Trihexyphenidyl", "DB01235": "Levodopa", "DB00313": "Valproic Acid",
    "DB00163": "Vitamin E", "DB00510": "Valproate", "DB00747": "Scopolamine",
    "DB00413": "Pramipexole", "DB00268": "Ropinirole", "DB00190": "Carbidopa",
    "DB00545": "Pyridostigmine", "DB01122": "Ambenonium", "DB00245": "Benztropine",
    "DB01400": "Neostigmine"
}


# =========================================================
# 🛠️ 辅助函数
# =========================================================
def get_name_mapping(csv_path, id_col_candidates, name_col_candidates):
    """尝试从 CSV 中提取 ID 到 真实名称 的映射字典"""
    df = pd.read_csv(csv_path)
    id_col = next((col for col in id_col_candidates if col in df.columns), df.columns[0])
    name_col = next((col for col in name_col_candidates if col in df.columns), id_col)
    return dict(zip(df[id_col].astype(str).str.strip(), df[name_col].astype(str).str.strip()))


def plot_tsne(embeddings, labels, title, save_path, highlight_idx, num_top_k):
    """绘制空间降维散点图 (使用 MDS 保留高维余弦距离)"""
    # 1. 计算余弦距离矩阵
    dist_matrix = cosine_distances(embeddings)

    # 2. MDS 降维到 2D 平面
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=3000)
    tsne_results = mds.fit_transform(dist_matrix)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    disease_x, disease_y = tsne_results[highlight_idx, 0], tsne_results[highlight_idx, 1]

    # 画所有药物点
    for i in range(len(embeddings)):
        if i == highlight_idx:
            continue

        is_top = i < num_top_k
        color = 'royalblue' if is_top else 'lightgray'
        marker = 'o' if is_top else 'X'
        size = 200 if is_top else 80
        alpha = 0.9 if is_top else 0.6

        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], c=color, marker=marker, s=size, alpha=alpha)

        if is_top:
            plt.annotate(labels[i], (tsne_results[i, 0], tsne_results[i, 1]),
                         xytext=(7, 7), textcoords='offset points', fontsize=12, fontweight='500')

    # 画疾病靶点（红星）
    plt.scatter(disease_x, disease_y, c='crimson', marker='*', s=800, edgecolor='black', linewidth=1.5, zorder=5)
    plt.annotate(labels[highlight_idx], (disease_x, disease_y), xytext=(12, -12), textcoords='offset points',
                 fontsize=18, fontweight='bold', color='crimson')

    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Dimension 1 (Semantic Distance)', fontsize=13)
    plt.ylabel('Dimension 2 (Semantic Distance)', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =========================================================
# 🚀 核心运行逻辑
# =========================================================
def main():
    print(f"\n{'=' * 60}")
    print(f"🔬 启动临床新药发现与空间可视化 (Case Study) - {CFG.DATASET_NAME}")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CFG.MODEL_WEIGHT_PATH):
        print(f"❌ 找不到模型权重文件: {CFG.MODEL_WEIGHT_PATH}")
        print("💡 请先运行训练脚本，保存一个 .pth 权重文件，然后修改 CFG.MODEL_WEIGHT_PATH。")
        return

    print("📦 正在加载图结构与文本信息...")
    mat = sio.loadmat(CFG.MAT_PATH)
    drug_sim = mat['drug'].astype(np.float32)
    disease_sim = mat['disease'].astype(np.float32)
    adj = mat['didr']

    if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
        adj = adj.T

    num_drugs, num_diseases = adj.shape
    ordered_drug_ids = [str(i[0][0]).strip() for i in mat['Wrname']]
    ordered_disease_ids = [str(i[0][0]).strip() for i in mat['Wdname']]

    drug_map = get_name_mapping(CFG.DRUG_CSV, ['db_id', 'drug_id'],
                                ['Name', 'name', 'Description', 'description', 'db_id'])
    disease_map = get_name_mapping(CFG.DISEASE_CSV, ['omim_id', 'disease_id'],
                                   ['Name', 'name', 'Description', 'description', 'omim_id'])

    dataset_path = os.path.join(CFG.PROC_DIR, "dataset.pt")
    data = torch.load(dataset_path, weights_only=False)
    adj_dd = torch.from_numpy(data['adj_dd']).float().to(device)
    adj_ss = torch.from_numpy(data['adj_ss']).float().to(device)
    adj_ds = torch.from_numpy(data['adj_ds']).float().to(device)

    d_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}drug_features.npy")).float().to(device)
    s_feat = torch.from_numpy(np.load(f"{CFG.FEAT_DIR}disease_features.npy")).float().to(device)

    # ---------------------------------------------------------
    # 🌟 1. 初始化并加载 真实模型 (包含专家知识)
    # ---------------------------------------------------------
    print(f"📥 加载预训练模型权重 (Expert Knowledge)...")
    model_real = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1], hidden_dim=CFG.HIDDEN_DIM,
        dropout=CFG.DROPOUT, edge_drop=0.0, num_drugs=num_drugs, num_diseases=num_diseases
    ).to(device)
    model_real.load_state_dict(torch.load(CFG.MODEL_WEIGHT_PATH, map_location=device))
    model_real.eval()

    with torch.no_grad():
        agg_d_real, agg_s_real = model_real(d_feat, s_feat, adj_dd, adj_ss, adj_ds)

    # ---------------------------------------------------------
    # 🎲 2. 初始化 随机模型 (用于生成图7中的对照组)
    # ---------------------------------------------------------
    print(f"🎲 初始化随机模型 (Random Initialization)...")
    d_feat_rand = torch.randn_like(d_feat).to(device)
    s_feat_rand = torch.randn_like(s_feat).to(device)
    model_rand = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1], hidden_dim=CFG.HIDDEN_DIM,
        dropout=0.0, edge_drop=0.0, num_drugs=num_drugs, num_diseases=num_diseases
    ).to(device)
    model_rand.eval()

    with torch.no_grad():
        agg_d_rand, agg_s_rand = model_rand(d_feat_rand, s_feat_rand, adj_dd, adj_ss, adj_ds)

    # =========================================================
    # 🔎 开始对每种目标疾病进行探索并画图
    # =========================================================
    all_results = []

    for kw in CFG.TARGET_DISEASES:
        target_dis_idx, target_dis_id, target_dis_name = -1, "", ""
        kw_lower = kw.lower()

        for idx, d_id in enumerate(ordered_disease_ids):
            # 解决 Unknown 问题
            d_id_clean = d_id.replace("D", "") if d_id.startswith("D") else d_id
            name = disease_map.get(d_id, disease_map.get(d_id_clean, "Unknown"))

            if kw_lower in name.lower() or kw_lower in d_id.lower() or kw_lower in d_id_clean.lower():
                target_dis_idx, target_dis_id, target_dis_name = idx, d_id, name
                break

        if target_dis_idx == -1:
            continue

        print(f"\n" + "=" * 50)
        print(f"🎯 正在分析疾病: {target_dis_name[:20]}... (ID: {target_dis_id})")
        print("=" * 50)

        unknown_drug_indices = np.where(adj[:, target_dis_idx] == 0)[0]
        test_x = torch.tensor([[d_idx, target_dis_idx] for d_idx in unknown_drug_indices]).long().to(device)

        with torch.no_grad():
            t_probs = torch.sigmoid(model_real.predict(agg_d_real, agg_s_real, test_x)).cpu().numpy()

        top_k_relative_idx = np.argsort(t_probs)[::-1][:CFG.TOP_K]
        top_k_drug_ids = []

        print(f"\n🏆 Top-{CFG.TOP_K} 潜在治疗药物推荐榜单:")
        print(f"{'Rank':<5} | {'Drug ID':<12} | {'Probability':<12} | {'Drug Name/Desc'}")
        print("-" * 80)

        for rank, rel_idx in enumerate(top_k_relative_idx):
            drug_idx = unknown_drug_indices[rel_idx]
            prob = t_probs[rel_idx]
            drug_id = ordered_drug_ids[drug_idx]
            drug_name = drug_map.get(drug_id, "Unknown")

            display_name = PROFESSIONAL_NAME_MAP.get(drug_id, drug_name)
            display_name = (display_name[:45] + '...') if len(display_name) > 45 else display_name

            print(f"{rank + 1:<5} | {drug_id:<12} | {prob:.6f}     | {display_name}")
            top_k_drug_ids.append(drug_id)
            all_results.append({
                "Disease": target_dis_name, "Disease_ID": target_dis_id,
                "Rank": rank + 1, "Drug_ID": drug_id, "Score": prob
            })

        # ---------------------------------------------------------
        # 🎨 开始绘制 图7 (特征空间聚类可视化)
        # ---------------------------------------------------------
        selected_drug_indices = []
        selected_labels = []

        # 抓取 Top K 药物
        for d_id in top_k_drug_ids:
            try:
                selected_drug_indices.append(ordered_drug_ids.index(d_id))
                selected_labels.append(PROFESSIONAL_NAME_MAP.get(d_id, d_id))
            except ValueError:
                pass

        # 抓取 背景噪音 药物
        for d_id in CFG.RANDOM_DRUGS:
            try:
                selected_drug_indices.append(ordered_drug_ids.index(d_id))
                selected_labels.append("Noise")
            except ValueError:
                pass

        highlight_disease_idx = len(selected_drug_indices)
        short_dis_name = "AD" if "104300" in target_dis_id else ("PD" if "168600" in target_dis_id else "Disease")
        selected_labels.append(short_dis_name)

        # 🚀 核心投影函数：处理对齐与归一化
        def get_plot_data(drug_embs, disease_embs, model_instance):
            plot_embs = drug_embs[selected_drug_indices].cpu().numpy()

            # 使用 bilinear_w 投影疾病，使之与药物空间对齐
            aligned_dis_emb = model_instance.bilinear_w(disease_embs[target_dis_idx]).detach()
            dis_emb = aligned_dis_emb.cpu().numpy().reshape(1, -1)

            # L2 余弦归一化
            plot_embs = plot_embs / (np.linalg.norm(plot_embs, axis=1, keepdims=True) + 1e-8)
            dis_emb = dis_emb / (np.linalg.norm(dis_emb, axis=1, keepdims=True) + 1e-8)

            return np.vstack([plot_embs, dis_emb])

        # 提取真实数据与随机数据
        real_data = get_plot_data(agg_d_real, agg_s_real, model_real)
        rand_data = get_plot_data(agg_d_rand, agg_s_rand, model_rand)

        print(f"\n🎨 正在生成 {short_dis_name} 的空间可视化图表...")

        # 绘制专家知识特征空间图
        plot_tsne(real_data, selected_labels, f"Semantic Space: Expert Knowledge ({short_dis_name})",
                  os.path.join(CFG.PROC_DIR, f"Fig7_expert_{target_dis_id}.png"), highlight_disease_idx, CFG.TOP_K)

        # 绘制随机初始化对照组图
        plot_tsne(rand_data, selected_labels, f"Semantic Space: Random Initialization ({short_dis_name})",
                  os.path.join(CFG.PROC_DIR, f"Fig7_random_{target_dis_id}.png"), highlight_disease_idx, CFG.TOP_K)

    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = os.path.join(CFG.PROC_DIR, "case_study_top10.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n💾 案例分析数据已保存至: {csv_path}")
        print("✅ 空间可视化图片已保存为以 Fig7_ 开头的 PNG 文件，请在 processed 文件夹中查收！")


if __name__ == "__main__":
    main()