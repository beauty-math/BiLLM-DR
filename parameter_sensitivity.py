import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from billm_dr.model import DReKGNN_PyTorch
from billm_dr.train_eval import BPR_BCE_Loss, get_knn_graph


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class SensitivityConfig:
    project_root: str = os.environ.get("PROJECT_ROOT", "/home/zqgaopengfei/project/zym")
    dataset_name: str = os.environ.get("DATASET_NAME", "Fdataset")
    seed: int = env_int("SEED", 42)
    hidden_dim: int = env_int("HIDDEN_DIM", 512)
    dropout: float = env_float("DROPOUT", 0.5)
    edge_drop: float = env_float("EDGE_DROP", 0.25)
    batch_size: int = env_int("BATCH_SIZE", 512)
    epochs: int = env_int("EPOCHS", 2000)
    patience: int = env_int("PATIENCE", 15)
    lr: float = env_float("LR", 1e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 5e-4)
    margin: float = env_float("MARGIN", 1.0)
    save_predictions: bool = env_bool("SAVE_PREDICTIONS", False)

    @property
    def data_root(self) -> str:
        return os.path.join(self.project_root, "data", self.dataset_name)

    @property
    def mat_path(self) -> str:
        return os.path.join(self.data_root, f"{self.dataset_name}.mat")

    @property
    def feat_dir(self) -> str:
        return os.path.join(self.data_root, "features")

    @property
    def result_dir(self) -> str:
        return os.environ.get(
            "RESULT_DIR",
            os.path.join(self.project_root, "results", self.dataset_name, "sensitivity"),
        )


CFG = SensitivityConfig()

BASE_PARAMS = {
    "neg_ratio": 90,
    "pos_weight": 12.0,
    "knn_k": 15,
    "bpr_weight": 0.3,
}

PARAMETER_SWEEP: List[Tuple[str, Iterable[float]]] = [
    ("neg_ratio", [20, 40, 60, 80, 90, 95]),
    ("pos_weight", [6, 8, 10, 12]),
    ("knn_k", [5, 10, 15, 20]),
    ("bpr_weight", [0.0, 0.1, 0.3, 0.5]),
]


def set_global_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mat_data(cfg: SensitivityConfig):
    mat = sio.loadmat(cfg.mat_path)
    drug_sim = mat["drug"].astype(np.float32)
    disease_sim = mat["disease"].astype(np.float32)
    adj = mat["didr"]
    if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
        adj = adj.T
    return drug_sim, disease_sim, adj.astype(np.float32)


def load_features(cfg: SensitivityConfig, device):
    drug_feat_path = os.path.join(cfg.feat_dir, "drug_features.npy")
    disease_feat_path = os.path.join(cfg.feat_dir, "disease_features.npy")
    if not os.path.exists(drug_feat_path) or not os.path.exists(disease_feat_path):
        raise FileNotFoundError(
            f"Missing feature files under {cfg.feat_dir}. Run feature extraction first."
        )
    d_feat = torch.from_numpy(np.load(drug_feat_path)).float().to(device)
    s_feat = torch.from_numpy(np.load(disease_feat_path)).float().to(device)
    return d_feat, s_feat


def make_split(adj: np.ndarray, seed: int, neg_ratio: int):
    pos_idx = np.argwhere(adj == 1)
    neg_idx_all = np.argwhere(adj == 0)

    train_pos, vt_pos = train_test_split(pos_idx, test_size=0.2, random_state=seed)
    val_pos, test_pos = train_test_split(vt_pos, test_size=0.5, random_state=seed)
    train_neg_all, vt_neg = train_test_split(neg_idx_all, test_size=0.2, random_state=seed)
    val_neg, test_neg = train_test_split(vt_neg, test_size=0.5, random_state=seed)

    np.random.seed(seed)
    sample_size = min(len(train_pos) * int(neg_ratio), len(train_neg_all))
    train_neg = train_neg_all[np.random.choice(len(train_neg_all), sample_size, replace=False)]

    train_x = np.vstack([train_pos, train_neg])
    train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    val_x = np.vstack([val_pos, val_neg])
    val_y = np.hstack([np.ones(len(val_pos)), np.zeros(len(val_neg))])
    test_x = np.vstack([test_pos, test_neg])
    test_y = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    train_adj_ds = np.zeros_like(adj, dtype=np.float32)
    for d, s in train_pos:
        train_adj_ds[d, s] = 1.0
    row_sum = train_adj_ds.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "adj_ds": train_adj_ds / row_sum,
        "train_pos": len(train_pos),
        "train_neg": len(train_neg),
        "val_pos": len(val_pos),
        "test_pos": len(test_pos),
    }


def evaluate(test_y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(test_y, probs),
        "AUPR": average_precision_score(test_y, probs),
        "Accuracy": accuracy_score(test_y, preds),
        "Precision": precision_score(test_y, preds, zero_division=0),
        "Recall": recall_score(test_y, preds, zero_division=0),
        "F1": f1_score(test_y, preds, zero_division=0),
    }


def train_single_run(
    cfg: SensitivityConfig,
    params: Dict[str, float],
    d_feat,
    s_feat,
    drug_sim: np.ndarray,
    disease_sim: np.ndarray,
    adj: np.ndarray,
):
    set_global_seed(cfg.seed)
    device = d_feat.device
    split = make_split(adj, cfg.seed, int(params["neg_ratio"]))

    adj_dd = torch.from_numpy(get_knn_graph(drug_sim.copy(), k=int(params["knn_k"]))).float().to(device)
    adj_ss = torch.from_numpy(get_knn_graph(disease_sim.copy(), k=int(params["knn_k"]))).float().to(device)
    adj_ds = torch.from_numpy(split["adj_ds"]).float().to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(split["train_x"]), torch.from_numpy(split["train_y"]).float()),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        edge_drop=cfg.edge_drop,
        num_drugs=adj.shape[0],
        num_diseases=adj.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = BPR_BCE_Loss(
        device=device,
        pos_weight=float(params["pos_weight"]),
        margin=cfg.margin,
        bpr_weight=float(params["bpr_weight"]),
        hard_negatives=False,
    )

    top_models = []
    best_aupr = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_x, batch_labels in train_loader:
            batch_x = batch_x.to(device).long()
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
            logits = model.predict(agg_d, agg_s, batch_x)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
                val_logits = model.predict(
                    agg_d, agg_s, torch.from_numpy(split["val_x"]).to(device).long()
                )
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_aupr = average_precision_score(split["val_y"], val_probs)

            if val_aupr > best_aupr:
                best_aupr = val_aupr
                patience_counter = 0
            else:
                patience_counter += 1

            state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if len(top_models) < 3:
                top_models.append((val_aupr, epoch, state_dict))
                top_models.sort(key=lambda x: x[0], reverse=True)
            elif val_aupr > top_models[-1][0]:
                top_models[-1] = (val_aupr, epoch, state_dict)
                top_models.sort(key=lambda x: x[0], reverse=True)

            if patience_counter >= cfg.patience:
                print(f"  early stop at epoch {epoch}; best val AUPR={best_aupr:.4f}")
                break

    test_x_tensor = torch.from_numpy(split["test_x"]).to(device).long()
    probs_total = np.zeros(len(split["test_y"]), dtype=np.float64)
    best_epochs = []
    for _, ep, state_dict in top_models:
        best_epochs.append(ep)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            agg_d, agg_s = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)
            logits = model.predict(agg_d, agg_s, test_x_tensor)
            probs_total += torch.sigmoid(logits).cpu().numpy()

    probs = probs_total / max(1, len(top_models))
    metrics = evaluate(split["test_y"], probs)
    metrics.update(
        {
            "Best_Val_AUPR": best_aupr,
            "Best_Epochs": ",".join(str(ep) for ep in best_epochs),
            "Train_Positives": split["train_pos"],
            "Train_Negatives": split["train_neg"],
            "Val_Positives": split["val_pos"],
            "Test_Positives": split["test_pos"],
        }
    )

    pred_df = None
    if cfg.save_predictions:
        pred_df = pd.DataFrame(
            {
                "Drug_ID": split["test_x"][:, 0].astype(int),
                "Disease_ID": split["test_x"][:, 1].astype(int),
                "True_Label": split["test_y"].astype(int),
                "Pred_Prob": probs,
            }
        )

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, pred_df


def build_runs() -> List[Dict[str, float]]:
    allowed = os.environ.get("ONLY_PARAM")
    allowed_params = None
    if allowed:
        allowed_params = {item.strip() for item in allowed.split(",") if item.strip()}

    runs = []
    run_id = 1
    for param_name, values in PARAMETER_SWEEP:
        if allowed_params and param_name not in allowed_params:
            continue
        for value in values:
            params = dict(BASE_PARAMS)
            params[param_name] = value
            runs.append(
                {
                    "Run_ID": run_id,
                    "Sweep_Parameter": param_name,
                    "Sweep_Value": value,
                    **params,
                }
            )
            run_id += 1
    return runs


def main():
    os.makedirs(CFG.result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{CFG.dataset_name}_single_sensitivity_seed{CFG.seed}_{timestamp}"

    print("Parameter sensitivity single-run experiment")
    print(f"Dataset: {CFG.dataset_name}")
    print(f"Result dir: {CFG.result_dir}")
    print(f"Epochs: {CFG.epochs}, patience: {CFG.patience}")
    print(f"Base params: {BASE_PARAMS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_feat, s_feat = load_features(CFG, device)
    drug_sim, disease_sim, adj = load_mat_data(CFG)
    runs = build_runs()
    print(f"Total runs: {len(runs)}")

    summary_rows = []
    for run in runs:
        params = {
            "neg_ratio": int(run["neg_ratio"]),
            "pos_weight": float(run["pos_weight"]),
            "knn_k": int(run["knn_k"]),
            "bpr_weight": float(run["bpr_weight"]),
        }
        print(
            f"\n[{run['Run_ID']:02d}/{len(runs)}] {run['Sweep_Parameter']}={run['Sweep_Value']} "
            f"| neg={params['neg_ratio']} posw={params['pos_weight']} "
            f"knn={params['knn_k']} bpr={params['bpr_weight']}"
        )
        metrics, pred_df = train_single_run(CFG, params, d_feat, s_feat, drug_sim, disease_sim, adj)
        row = {**run, **metrics}
        summary_rows.append(row)
        print(f"  AUC={metrics['AUC']:.4f} AUPR={metrics['AUPR']:.4f} F1={metrics['F1']:.4f}")

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(CFG.result_dir, f"{run_tag}_summary_partial.csv"), index=False
        )

        if pred_df is not None:
            pred_name = (
                f"{run_tag}_run{run['Run_ID']:02d}_{run['Sweep_Parameter']}"
                f"_{str(run['Sweep_Value']).replace('.', 'p')}_predictions.csv"
            )
            pred_df.to_csv(os.path.join(CFG.result_dir, pred_name), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(CFG.result_dir, f"{run_tag}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    config_path = os.path.join(CFG.result_dir, f"{run_tag}_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"dataset_name: {CFG.dataset_name}\n")
        f.write(f"seed: {CFG.seed}\n")
        f.write(f"epochs: {CFG.epochs}\n")
        f.write(f"patience: {CFG.patience}\n")
        f.write(f"base_params: {BASE_PARAMS}\n")
        f.write(f"parameter_sweep: {PARAMETER_SWEEP}\n")
        f.write(f"save_predictions: {CFG.save_predictions}\n")

    print("\nParameter sensitivity completed.")
    print(f"Summary saved to: {summary_path}")
    print("\nMean table:")
    print(
        summary_df[
            [
                "Run_ID",
                "Sweep_Parameter",
                "Sweep_Value",
                "neg_ratio",
                "pos_weight",
                "knn_k",
                "bpr_weight",
                "AUC",
                "AUPR",
                "F1",
            ]
        ].to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
