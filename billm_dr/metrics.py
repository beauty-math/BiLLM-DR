import numpy as np
from sklearn.metrics import ndcg_score
from scipy import stats


def calculate_ranking_metrics(y_true, y_pred, k=50):
    """
    计算单个查询（例如一个疾病对应的所有药物预测）的 Recall@K 和 NDCG@K

    参数:
    y_true: 真实标签的 1D numpy 数组 (1 表示有关联, 0 表示无关联)
    y_pred: 预测概率的 1D numpy 数组
    k: 截断位置 (Top K)

    返回:
    recall_at_k, ndcg_at_k
    """
    num_positive = np.sum(y_true)

    # 如果真实标签中没有正样本，则不计入该查询的统计
    if num_positive == 0:
        return 0.0, 0.0

        # 获取预测概率最高的 K 个索引
    top_k_indices = np.argsort(y_pred)[::-1][:k]

    # 计算 Recall@K
    hits = np.sum(y_true[top_k_indices])
    recall_at_k = hits / num_positive

    # 计算 NDCG@K (sklearn 需要 2D 数组)
    ndcg_at_k = ndcg_score([y_true], [y_pred], k=k)

    return recall_at_k, ndcg_at_k


def run_significance_tests(our_scores, baseline_scores, metric_name="AUC"):
    """
    进行配对 t 检验和 Wilcoxon 符号秩检验
    (此函数可在您跑完基线模型后，手动传入得分列表进行统计检验)

    参数:
    our_scores: 我们模型的得分列表 (例如 10 折的 AUC 列表)
    baseline_scores: 基线模型的得分列表 (长度必须与 our_scores 相同)
    metric_name: 打印输出时使用的指标名称
    """
    if len(our_scores) != len(baseline_scores):
        print("错误：两组得分的长度不一致，无法进行配对检验。")
        return

    print(f"\n=== 统计显著性检验 ({metric_name}) ===")
    print(f"Our Model 平均得分: {np.mean(our_scores):.4f}")
    print(f"Baseline  平均得分: {np.mean(baseline_scores):.4f}")

    # 配对 t 检验
    t_stat, p_val_t = stats.ttest_rel(our_scores, baseline_scores)

    # Wilcoxon 符号秩检验
    w_stat, p_val_w = stats.wilcoxon(our_scores, baseline_scores)

    print("-" * 30)
    print(f"配对 t 检验 (Paired t-test) p-value:   {p_val_t:.3e}")
    print(f"Wilcoxon 符号秩检验 p-value:          {p_val_w:.3e}")

    if p_val_w < 0.05:
        print("✅ 结论: 你的模型显著优于基线模型 (p < 0.05)")
    else:
        print("⚠️ 结论: 两个模型差异不具有统计学显著性")