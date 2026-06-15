from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


PROJECT = Path("/home/zqgaopengfei/project/zym")
OUT = PROJECT / "results" / "paper_result_figures"
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "Cdataset": "#4C78A8",
    "Fdataset": "#F58518",
    "known": "#1FA65A",
    "candidate": "#2F6BDE",
    "target": "#D62728",
    "background": "#DDE3EA",
}


DISEASE_NAMES = {
    "168600": "Parkinson disease",
    "600807": "Asthma",
    "157300": "Migraine",
    "125853": "Type 2 diabetes",
    "165720": "Osteoarthritis",
    "181500": "Schizophrenia",
}


def setup_style():
    font_candidates = [
        PROJECT / "results/performance_plots/fonts/times.ttf",
        PROJECT / "results/performance_plots/fonts/Times New Roman.ttf",
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    ]
    for fp in font_candidates:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            break
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })


def savefig(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def read_topk():
    paths = {
        "Cdataset": PROJECT / "results/Cdataset/topk/Cdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_164248_topk_20260512_032242_summary.csv",
        "Fdataset": PROJECT / "results/Fdataset/topk/Fdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_110357_topk_20260512_032242_summary.csv",
    }
    rows = []
    for dataset, path in paths.items():
        df = pd.read_csv(path)
        df["Dataset"] = dataset
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "figure5_topk_summary.csv", index=False)
    return out


def fig_topk():
    df = read_topk()
    metrics = [
        ("Precision@K_mean", "Precision@K_std", "Precision@K", (0.0, 1.08)),
        ("Recall@K_mean", "Recall@K_std", "Recall@K", (0.0, 1.08)),
        ("NDCG@K_mean", "NDCG@K_std", "NDCG@K", (0.0, 1.08)),
    ]
    ks = [10, 20, 50, 6000]
    x = np.arange(len(ks))
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.65), sharex=True)
    for ax, (mean_col, std_col, ylabel, ylim) in zip(axes, metrics):
        for dataset in ["Cdataset", "Fdataset"]:
            sub = df[df["Dataset"] == dataset].set_index("K").loc[ks]
            ax.errorbar(
                x, sub[mean_col], yerr=sub[std_col],
                marker="o", markersize=3.8, linewidth=1.45, capsize=2.0,
                color=COLORS[dataset], label=dataset
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7.5)
        ax.set_xlabel("K", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7.5)
    axes[0].text(-0.18, 1.02, "A", transform=axes[0].transAxes, fontsize=11.5, fontweight="bold")
    axes[1].text(-0.18, 1.02, "B", transform=axes[1].transAxes, fontsize=11.5, fontweight="bold")
    axes[2].text(-0.18, 1.02, "C", transform=axes[2].transAxes, fontsize=11.5, fontweight="bold")
    handles = [
        Line2D([0], [0], color=COLORS["Cdataset"], marker="o", linewidth=1.45, markersize=3.8, label="Cdataset"),
        Line2D([0], [0], color=COLORS["Fdataset"], marker="o", linewidth=1.45, markersize=3.8, label="Fdataset"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(w_pad=1.3)
    savefig(fig, "figure5_topk_recommendation")


def read_case_detail():
    path = PROJECT / "results/case_study_verification/all_selected_diseases/all_selected_case_study_evidence_detail.csv"
    df = pd.read_csv(path)
    df["OMIM"] = df["OMIM"].astype(str)
    df["Evidence_supported"] = df["Pieces_of_evidence"].fillna("NA").ne("NA")
    df["Disease_short"] = df["OMIM"].map(DISEASE_NAMES).fillna(df["Disease"])
    return df


def fig_case_study():
    df = read_case_detail()
    supported = (
        df.groupby(["Disease_short", "Dataset"], as_index=False)
        .agg(Supported=("Evidence_supported", "sum"), Total=("Evidence_supported", "size"))
    )
    supported["Support_rate"] = supported["Supported"] / supported["Total"]
    supported.to_csv(OUT / "figure6_case_study_evidence_summary.csv", index=False)

    order = (
        supported.groupby("Disease_short")["Support_rate"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    height = 0.34
    for idx, dataset in enumerate(["Cdataset", "Fdataset"]):
        sub = supported[supported["Dataset"] == dataset].set_index("Disease_short").reindex(order)
        vals = sub["Support_rate"].values
        counts = sub["Supported"].values
        ax.barh(
            y + (idx - 0.5) * height, vals, height,
            color=COLORS[dataset], edgecolor="#1f2937", linewidth=0.55,
            label=dataset
        )
        for yi, val, count in zip(y + (idx - 0.5) * height, vals, counts):
            ax.text(min(val + 0.018, 0.98), yi, f"{int(count)}/10",
                    va="center", ha="left", fontsize=7.3)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Evidence-supported proportion in top-10 candidates", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "figure6_case_study_evidence")

    table = (
        df.sort_values(["Disease_short", "Dataset", "Rank"])
        [["Dataset", "OMIM", "Disease_short", "Rank", "Drug", "DrugBank_ID", "Prediction_score", "Pieces_of_evidence"]]
    )
    table.to_csv(OUT / "table_case_study_top10_all_selected.csv", index=False)

    selected = df[
        (df["OMIM"].isin(["600807", "157300", "165720"]))
        & (df["Rank"] <= 10)
    ].copy()
    selected.to_csv(OUT / "table_case_study_top10_selected_main.csv", index=False)


def load_query_points(dataset, omim):
    path = PROJECT / f"results/{dataset}/visualization_query_space/{dataset}_target_{omim}_query_space_points.csv"
    df = pd.read_csv(path)
    df["OMIM"] = str(omim)
    return df


def fig_visualization():
    targets = ["168600", "600807", "157300"]
    datasets = ["Cdataset", "Fdataset"]
    all_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6), sharex=False, sharey=False)
    for r, dataset in enumerate(datasets):
        for c, omim in enumerate(targets):
            ax = axes[r, c]
            df = load_query_points(dataset, omim)
            all_rows.append(df.assign(Dataset=dataset))
            for role, marker, size, color, zorder in [
                ("Background nearest drugs", "o", 16, COLORS["background"], 1),
                ("Known associated drug", "o", 42, COLORS["known"], 3),
                ("Top predicted candidate", "^", 52, COLORS["candidate"], 4),
                ("Target disease query", "*", 140, COLORS["target"], 5),
            ]:
                sub = df[df["Role"] == role]
                if len(sub) == 0:
                    continue
                ax.scatter(sub["X"], sub["Y"], s=size, marker=marker, color=color,
                           alpha=0.88 if role != "Background nearest drugs" else 0.42,
                           edgecolor="white" if role != "Background nearest drugs" else "none",
                           linewidth=0.35, zorder=zorder)
            label_df = df[df["Role"].isin(["Known associated drug", "Top predicted candidate"])].copy()
            label_df = label_df.sort_values("Pred_Prob", ascending=False).head(4)
            for _, row in label_df.iterrows():
                ax.text(row["X"], row["Y"], str(row["Drug_ID"]), fontsize=5.8,
                        ha="left", va="bottom")
            ax.axhline(0, color="#E5E7EB", linewidth=0.8, zorder=0)
            ax.axvline(0, color="#E5E7EB", linewidth=0.8, zorder=0)
            ax.set_title(f"{dataset}, {DISEASE_NAMES[omim]}", fontsize=8.8, pad=3)
            if r == 1:
                ax.set_xlabel("Query-space PC1", fontsize=8.3)
            if c == 0:
                ax.set_ylabel("Query-space PC2", fontsize=8.3)
            ax.tick_params(labelsize=6.8)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT / "figure7_query_space_points.csv", index=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["background"], markersize=6, label="Background"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["known"], markersize=7, label="Known associated drug"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["candidate"], markersize=7, label="Top predicted candidate"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["target"], markersize=10, label="Target disease query"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=1.0, h_pad=1.1)
    savefig(fig, "figure7_query_space_visualization")


def write_results_text():
    topk = read_topk()
    case_df = read_case_detail()
    evidence = (
        case_df.groupby(["Dataset"])["Evidence_supported"]
        .agg(["sum", "count"])
        .reset_index()
    )
    evidence["rate"] = evidence["sum"] / evidence["count"]

    def get_topk(dataset, k, metric):
        row = topk[(topk["Dataset"] == dataset) & (topk["K"] == k)].iloc[0]
        return row[f"{metric}_mean"], row[f"{metric}_std"]

    c_p10, c_p10s = get_topk("Cdataset", 10, "Precision@K")
    f_p10, f_p10s = get_topk("Fdataset", 10, "Precision@K")
    c_r6000, c_r6000s = get_topk("Cdataset", 6000, "Recall@K")
    f_r6000, f_r6000s = get_topk("Fdataset", 6000, "Recall@K")
    c_n6000, c_n6000s = get_topk("Cdataset", 6000, "NDCG@K")
    f_n6000, f_n6000s = get_topk("Fdataset", 6000, "NDCG@K")

    c_rate = evidence[evidence["Dataset"] == "Cdataset"]["rate"].iloc[0]
    f_rate = evidence[evidence["Dataset"] == "Fdataset"]["rate"].iloc[0]

    text = f"""## Top-K 推荐性能分析

药物重定位任务不仅要求模型能够区分已知关联与未知关联，更重要的是从大规模候选药物-疾病组合中优先筛选出少量最有潜力的候选。因此，本文进一步采用 Precision@K、Recall@K 和 NDCG@K 评价 BiLLM-DR 的推荐排序能力，结果如图 5 所示。整体来看，模型在较小 K 值下保持了较高的命中精度，在较大 K 值下则能够覆盖绝大多数真实正关联，说明其兼具高置信候选筛选能力和大范围候选召回能力。

在 Top-10 推荐场景中，BiLLM-DR 在 Cdataset 上的 Precision@10 达到 {c_p10:.4f} ± {c_p10s:.4f}，在 Fdataset 上达到 {f_p10:.4f} ± {f_p10s:.4f}，表明模型排序最靠前的候选药物中大部分为真实关联样本。随着 K 从 10 增加到 50，Precision@K 略有下降，而 Recall@K 持续升高，这符合推荐系统评价中的典型规律：候选列表扩大后能够覆盖更多真实正样本，但同时也会引入更多低置信候选。值得注意的是，在 K=6000 时，Cdataset 和 Fdataset 的 Recall@6000 分别达到 {c_r6000:.4f} ± {c_r6000s:.4f} 和 {f_r6000:.4f} ± {f_r6000s:.4f}，说明模型能够在较大候选范围内召回超过 96% 的真实关联。

NDCG@K 进一步反映真实关联在排序列表中的位置质量。Cdataset 和 Fdataset 在 K=6000 时的 NDCG@6000 分别为 {c_n6000:.4f} ± {c_n6000s:.4f} 和 {f_n6000:.4f} ± {f_n6000s:.4f}，说明即使在大规模候选筛选条件下，真实关联也倾向于被排在相对靠前的位置。上述结果与 AUPR 的提升趋势一致，进一步证明 BiLLM-DR 具有较强的候选药物优先排序能力。

**图 5. BiLLM-DR 的 Top-K 推荐性能。**  
(A) Precision@K；(B) Recall@K；(C) NDCG@K。K 取 10、20、50 和 6000，误差线表示十折交叉验证的标准差。

## 药物重定位案例验证

为了进一步评估 BiLLM-DR 预测结果的生物医学合理性，本文针对多个代表性疾病开展药物重定位案例验证。具体而言，选取哮喘、偏头痛、2 型糖尿病、精神分裂症和骨关节炎等疾病，分别在 Cdataset 和 Fdataset 上提取预测得分最高的 Top-10 候选药物，并通过 DrugBank、PubChem、DrugCentral 和 ClinicalTrials.gov 等公开数据库进行证据检索。由于 CTD 在自动访问过程中受限，本文未将未人工确认的 CTD 结果计入证据表，以保证案例验证的保守性和可追溯性。

图 6 展示了不同疾病 Top-10 候选药物中具有外部证据支持的比例。总体上，Cdataset 和 Fdataset 的 Top-10 候选药物均包含一定数量已被数据库或临床试验记录支持的候选，证据支持比例分别为 {c_rate:.2%} 和 {f_rate:.2%}。例如，在哮喘相关疾病中，模型优先推荐了 Prednisone、Zafirlukast、Dexamethasone、Ciclesonide 和 Methylprednisolone 等药物，这些药物与抗炎或支气管哮喘治疗具有明确关联；在偏头痛案例中，Ibuprofen、Metoprolol、Valproic acid 和 Diclofenac 等候选药物具有药物说明或文献数据库支持；在骨关节炎案例中，Nabumetone、Acetaminophen、Salsalate、Prednisone 和 Meclofenamic acid 等候选药物也获得了外部证据支持。

这些结果表明，BiLLM-DR 的高分候选不仅来自统计意义上的关联模式，也能够对应到已有药理学知识、适应症记录或临床试验信息。与此同时，部分高排名候选暂未在所检索数据库中获得直接证据，可能代表潜在的新型重定位假设，也可能反映模型预测中的噪声。因此，本文将其标记为 NA，并保留为后续文献检索、分子机制分析或实验验证的候选对象。

**图 6. 不同疾病 Top-10 候选药物的外部证据支持比例。**  
柱旁数字表示 Top-10 候选药物中获得 DrugBank、PubChem、DrugCentral 或 ClinicalTrials.gov 等数据库支持的数量。NA 表示在当前检索数据库中未发现直接证据。

## 可视化分析

为直观展示 BiLLM-DR 学到的疾病查询空间与候选药物排序之间的关系，本文进一步开展 query-space 可视化分析。与直接在原始药物和疾病嵌入空间中计算二维投影不同，query-space 可视化以目标疾病为查询中心，将已知关联药物、高分预测候选药物以及背景邻近药物投影到以目标疾病为中心的局部表示空间中。该方式更符合双线性预测器的使用方式，因为模型最终关注的是药物表示与特定疾病查询之间的匹配关系，而不是简单的全局欧氏距离。

图 7 展示了 Parkinson disease、Asthma 和 Migraine 三个目标疾病在 Cdataset 和 Fdataset 上的 query-space 分布。可以观察到，已知关联药物和高分预测候选药物通常聚集在目标疾病查询点附近，且明显区别于大量背景邻近药物。例如，在 Parkinson disease 的可视化中，Ropinirole、Entacapone、Pramipexole、Rivastigmine 等已知或高度相关的抗帕金森药物位于目标查询附近，高分候选药物也分布在相近区域。Asthma 和 Migraine 的可视化结果同样显示，模型推荐的候选药物与已知关联药物在疾病查询空间中具有较高局部一致性。

该结果从表示学习角度说明，BiLLM-DR 并非仅依赖单一相似度或局部邻居进行预测，而是能够通过 LLM 语义特征、异构图传播和双线性疾病查询共同形成面向目标疾病的候选药物排序空间。已知关联药物与高分预测候选在 query-space 中的相对聚集，为模型的排序输出提供了直观解释。

**图 7. BiLLM-DR 的目标疾病 query-space 可视化分析。**  
红色星形表示目标疾病查询，绿色圆点表示已知关联药物，蓝色三角形表示 Top predicted candidates，浅灰色点表示背景邻近药物。每个子图以目标疾病为中心展示局部候选药物分布。
"""
    (OUT / "additional_results_sections_topk_case_visualization.md").write_text(text, encoding="utf-8")


def main():
    setup_style()
    fig_topk()
    fig_case_study()
    fig_visualization()
    write_results_text()
    print(f"Saved additional figures and text to {OUT}")


if __name__ == "__main__":
    main()
