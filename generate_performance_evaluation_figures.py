from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


PROJECT = Path("/home/zqgaopengfei/project/zym")
OUT = PROJECT / "results" / "paper_result_figures"
OUT.mkdir(parents=True, exist_ok=True)


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
        "font.size": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


COLORS = {
    "Cdataset": "#4C78A8",
    "Fdataset": "#F58518",
}


def savefig(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def read_summary(dataset):
    if dataset == "Cdataset":
        path = PROJECT / "results/Cdataset/10fold_cv/Cdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_164248_metrics_summary.csv"
    else:
        path = PROJECT / "results/Fdataset/Fdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_110357_metrics_summary.csv"
    df = pd.read_csv(path)
    df["Dataset"] = dataset
    return df


def read_fold_metrics(dataset):
    if dataset == "Cdataset":
        path = PROJECT / "results/Cdataset/10fold_cv/Cdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_164248_fold_metrics.csv"
    else:
        path = PROJECT / "results/Fdataset/Fdataset_10fold_seed42_neg90_posw12.0_bpr0.3_knn15_20260511_110357_fold_metrics.csv"
    df = pd.read_csv(path)
    df["Dataset"] = dataset
    return df


def normalize_ablation_name(name):
    name = name.replace("w_o_", "w/o_")
    return name


def read_ablation(dataset):
    base = PROJECT / f"results/{dataset}/ablations"
    rows = []
    # Start from the consolidated summary when available. It is the most reliable
    # source for Cdataset because some ablation names contain "w/o".
    for path in sorted(base.glob("*_all_metrics_summary.csv")):
        df = pd.read_csv(path)
        df["Ablation"] = df["Ablation"].astype(str).map(normalize_ablation_name)
        df["Dataset"] = dataset
        rows.append(df)
    # Then supplement individual summaries, including files stored under an
    # accidental "w/" directory caused by names such as "w/o_BPR_Loss".
    for path in sorted(base.rglob("*_metrics_summary.csv")):
        if "_all_metrics_summary" in path.name:
            continue
        df = pd.read_csv(path)
        if "Ablation" not in df.columns:
            m = re.search(r"_(Ours_Full|w[_/]?o_[A-Za-z0-9_]+)_metrics_summary", str(path))
            if m:
                df["Ablation"] = normalize_ablation_name(m.group(1))
            else:
                continue
        df["Ablation"] = df["Ablation"].astype(str).map(normalize_ablation_name)
        df["Dataset"] = dataset
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    # Remove possible duplicates by keeping the newest/last row order per dataset-ablation-metric.
    out = out.drop_duplicates(["Dataset", "Ablation", "Metric"], keep="last")
    return out


def read_sensitivity(dataset):
    if dataset == "Cdataset":
        path = PROJECT / "results/Cdataset/sensitivity/Cdataset_single_sensitivity_seed42_20260512_012505_summary.csv"
    else:
        path = PROJECT / "results/Fdataset/sensitivity/Fdataset_single_sensitivity_seed42_20260512_031302_summary.csv"
    df = pd.read_csv(path)
    df["Dataset"] = dataset
    return df


def read_seed(dataset):
    if dataset == "Cdataset":
        path = PROJECT / "results/Cdataset/seed_stability/Cdataset_seed_stability_neg90_posw12.0_bpr0.3_knn15_20260512_033623_per_seed.csv"
    else:
        path = PROJECT / "results/Fdataset/seed_stability/Fdataset_seed_stability_neg90_posw12.0_bpr0.3_knn15_20260512_033623_per_seed.csv"
    df = pd.read_csv(path)
    df["Dataset"] = dataset
    return df


def read_baseline_performance():
    path = PROJECT / "性能" / "performance_cleaned_for_plot.csv"
    df = pd.read_csv(path)
    df["Method"] = df["Method"].replace({"BiLLM-DR": "BiLLM-DR (Ours)"})
    rows = []
    for _, row in df.iterrows():
        for dataset in ["Cdataset", "Fdataset"]:
            for metric in ["AUC", "AUPR"]:
                rows.append({
                    "Method": row["Method"],
                    "Year": row["Year"],
                    "Dataset": dataset,
                    "Metric": metric,
                    "Mean": row[f"{dataset}_{metric}"],
                    "Std": row[f"{dataset}_{metric}_std"],
                    "Is_Ours": row["Method"] == "BiLLM-DR (Ours)",
                })
    out = pd.DataFrame(rows)
    out["Rank"] = out.groupby(["Dataset", "Metric"])["Mean"].rank(method="min", ascending=False)
    return out


def fig_10fold():
    perf = read_baseline_performance()
    perf.to_csv(OUT / "figure1_baseline_comparison_10fold.csv", index=False)

    summary = pd.concat([read_summary("Cdataset"), read_summary("Fdataset")], ignore_index=True)
    folds = pd.concat([read_fold_metrics("Cdataset"), read_fold_metrics("Fdataset")], ignore_index=True)
    summary.to_csv(OUT / "figure1_10fold_summary.csv", index=False)
    folds.to_csv(OUT / "figure1_10fold_fold_metrics.csv", index=False)

    order = (
        perf[perf["Metric"] == "AUPR"]
        .groupby("Method")["Mean"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    y = np.arange(len(order))
    panels = [
        ("Cdataset", "AUC", "A", (0.80, 0.985)),
        ("Cdataset", "AUPR", "B", (0.18, 0.735)),
        ("Fdataset", "AUC", "C", (0.78, 0.985)),
        ("Fdataset", "AUPR", "D", (0.10, 0.655)),
    ]
    dataset_base = {"Cdataset": "#B8C9DD", "Fdataset": "#E8C98F"}
    dataset_ours = {"Cdataset": "#B91C1C", "Fdataset": "#D97706"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    for ax, (dataset, metric, letter, xlim) in zip(axes.flat, panels):
        sub = (
            perf[(perf["Dataset"] == dataset) & (perf["Metric"] == metric)]
            .sort_values("Mean", ascending=True)
            .reset_index(drop=True)
        )
        y_pos = np.arange(len(sub))
        colors = []
        edges = []
        widths = []
        for _, row in sub.iterrows():
            if row["Is_Ours"]:
                colors.append(dataset_ours[dataset])
                edges.append("#111827")
                widths.append(0.85)
            elif row["Rank"] == 1:
                colors.append("#315A99")
                edges.append("#111827")
                widths.append(0.65)
            else:
                colors.append(dataset_base[dataset])
                edges.append("#334155")
                widths.append(0.45)
        ax.barh(y_pos, sub["Mean"], xerr=sub["Std"].fillna(0), height=0.58,
                color=colors, edgecolor=edges, linewidth=widths, capsize=2.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["Method"], fontsize=8.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel(metric, fontsize=10.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.text(-0.10, 1.03, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.text(0.50, 1.03, dataset, transform=ax.transAxes, fontsize=10,
                ha="center", va="bottom")
    handles = [
        Patch(facecolor="#B91C1C", edgecolor="#111827", label="BiLLM-DR on Cdataset"),
        Patch(facecolor="#D97706", edgecolor="#111827", label="BiLLM-DR on Fdataset"),
        Patch(facecolor="#315A99", edgecolor="#111827", label="Best baseline"),
        Patch(facecolor="#D7DEE8", edgecolor="#334155", label="Other baselines"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.50, 1.015), columnspacing=1.4, handlelength=1.5, fontsize=9)
    fig.tight_layout(w_pad=1.7, h_pad=1.8, rect=[0, 0, 1, 0.955])
    savefig(fig, "figure1_10fold_cross_validation")


def fig_ablation():
    df = pd.concat([read_ablation("Cdataset"), read_ablation("Fdataset")], ignore_index=True)
    df.to_csv(OUT / "figure2_ablation_summary.csv", index=False)
    order = [
        "Ours_Full", "w/o_Text_Feat", "w/o_DrugSim", "w/o_DiseaseSim",
        "w/o_BPR_Loss", "w/o_MultiHop_Weights", "w/o_SWA", "w/o_EdgeDrop"
    ]
    labels = {
        "Ours_Full": "Full",
        "w/o_Text_Feat": "w/o Text",
        "w/o_DrugSim": "w/o DrugSim",
        "w/o_DiseaseSim": "w/o DiseaseSim",
        "w/o_BPR_Loss": "w/o Ranking",
        "w/o_MultiHop_Weights": "w/o HopWeights",
        "w/o_SWA": "w/o Ensemble",
        "w/o_EdgeDrop": "w/o EdgeDrop",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    for ax, metric, xlim in zip(axes, ["AUPR", "AUC"], [(0.45, 0.74), (0.88, 0.99)]):
        y = np.arange(len(order))
        height = 0.34
        for idx, ds in enumerate(["Cdataset", "Fdataset"]):
            sub = df[(df["Dataset"] == ds) & (df["Metric"] == metric)].set_index("Ablation")
            vals = [sub.loc[o, "Mean"] if o in sub.index else np.nan for o in order]
            stds = [sub.loc[o, "Std"] if o in sub.index else 0 for o in order]
            ax.barh(y + (idx - 0.5) * height, vals, height, xerr=stds, capsize=2,
                    color=COLORS[ds], edgecolor="#1f2937", linewidth=0.5, alpha=0.88, label=ds)
        ax.set_xlim(*xlim)
        ax.set_xlabel(metric)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[o] for o in order])
        ax.invert_yaxis()
        ax.axvline(df[(df["Ablation"] == "Ours_Full") & (df["Metric"] == metric) & (df["Dataset"] == "Cdataset")]["Mean"].iloc[0],
                   color=COLORS["Cdataset"], linestyle=":", linewidth=1)
        ax.axvline(df[(df["Ablation"] == "Ours_Full") & (df["Metric"] == metric) & (df["Dataset"] == "Fdataset")]["Mean"].iloc[0],
                   color=COLORS["Fdataset"], linestyle=":", linewidth=1)
    handles = [
        Patch(facecolor=COLORS["Cdataset"], edgecolor="#1f2937", label="Cdataset"),
        Patch(facecolor=COLORS["Fdataset"], edgecolor="#1f2937", label="Fdataset"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.50, 1.02), columnspacing=1.8, handlelength=1.8)
    axes[0].text(-0.20, 1.03, "A", transform=axes[0].transAxes, fontsize=15, fontweight="bold")
    axes[1].text(-0.12, 1.03, "B", transform=axes[1].transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.93])
    savefig(fig, "figure2_ablation_study")


def fig_sensitivity():
    df = pd.concat([read_sensitivity("Cdataset"), read_sensitivity("Fdataset")], ignore_index=True)
    df.to_csv(OUT / "figure3_parameter_sensitivity.csv", index=False)
    params = [
        ("neg_ratio", "Negative ratio"),
        ("pos_weight", "Positive weight"),
        ("knn_k", "KNN k"),
        ("bpr_weight", "Ranking weight"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.1), sharey="row")
    for col, (param, label) in enumerate(params):
        for row, metric in enumerate(["AUPR", "AUC"]):
            ax = axes[row, col]
            for ds in ["Cdataset", "Fdataset"]:
                sub = df[(df["Dataset"] == ds) & (df["Sweep_Parameter"] == param)].sort_values("Sweep_Value")
                ax.plot(sub["Sweep_Value"], sub[metric], marker="o", linewidth=1.8, markersize=4.5,
                        color=COLORS[ds], label=ds)
            ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(metric)
            if row == 0:
                ax.set_ylim(0.50, 0.76)
            else:
                ax.set_ylim(0.955, 0.989)
            if col == 0 and row == 0:
                ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0, 0].text(-0.30, 1.08, "A", transform=axes[0, 0].transAxes, fontsize=15, fontweight="bold")
    axes[1, 0].text(-0.30, 1.08, "B", transform=axes[1, 0].transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    savefig(fig, "figure3_parameter_sensitivity")


def fig_seed():
    df = pd.concat([read_seed("Cdataset"), read_seed("Fdataset")], ignore_index=True)
    df.to_csv(OUT / "figure4_seed_stability.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, metric, ylim in zip(axes, ["AUPR", "AUC"], [(0.55, 0.75), (0.95, 0.99)]):
        data = [df[df["Dataset"] == ds][metric].values for ds in ["Cdataset", "Fdataset"]]
        bp = ax.boxplot(data, positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False)
        for patch, ds in zip(bp["boxes"], ["Cdataset", "Fdataset"]):
            patch.set_facecolor(COLORS[ds])
            patch.set_alpha(0.35)
            patch.set_edgecolor("#1f2937")
        rng = np.random.default_rng(7)
        for pos, ds, values in zip([1, 2], ["Cdataset", "Fdataset"], data):
            ax.scatter(np.full(len(values), pos) + rng.normal(0, 0.035, len(values)),
                       values, s=28, color=COLORS[ds], edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(pos, max(values) + (ylim[1] - ylim[0]) * 0.045,
                    f"{np.mean(values):.3f}±{np.std(values):.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Cdataset", "Fdataset"])
        ax.set_ylabel(metric)
        ax.set_ylim(*ylim)
    axes[0].text(-0.16, 1.04, "A", transform=axes[0].transAxes, fontsize=15, fontweight="bold")
    axes[1].text(-0.16, 1.04, "B", transform=axes[1].transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    savefig(fig, "figure4_random_seed_stability")


def main():
    setup_style()
    fig_10fold()
    fig_ablation()
    fig_sensitivity()
    fig_seed()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
