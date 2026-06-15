import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from billm_dr.model import DReKGNN_PyTorch


PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/zqgaopengfei/project/zym")
DATASETS = [
    item.strip()
    for item in os.environ.get("DATASET_NAME", os.environ.get("DATASETS", "Fdataset,Cdataset")).split(",")
    if item.strip()
]
TARGET_DISEASES = [
    item.strip()
    for item in os.environ.get("TARGET_DISEASES", "104300,168600").split(",")
    if item.strip()
]
TOP_K = int(os.environ.get("TOP_K", 20))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 512))
DROPOUT = float(os.environ.get("DROPOUT", 0.5))


def normalize_id(value: str) -> str:
    value = str(value).strip()
    return value[1:] if value.upper().startswith("D") and value[1:].isdigit() else value


def load_name_maps(drug_csv: str, disease_csv: str):
    drug_df = pd.read_csv(drug_csv)
    disease_df = pd.read_csv(disease_csv)

    drug_id_col = "db_id" if "db_id" in drug_df.columns else drug_df.columns[0]
    drug_name_col = "Name" if "Name" in drug_df.columns else "Description"
    if drug_name_col not in drug_df.columns:
        drug_name_col = drug_id_col

    disease_id_col = "omim_id" if "omim_id" in disease_df.columns else disease_df.columns[0]
    disease_name_col = "Name" if "Name" in disease_df.columns else "Description"
    if disease_name_col not in disease_df.columns:
        disease_name_col = disease_id_col

    drug_map = dict(
        zip(drug_df[drug_id_col].astype(str).str.strip(), drug_df[drug_name_col].astype(str).str.strip())
    )
    disease_map = {}
    disease_desc_map = {}
    for _, row in disease_df.iterrows():
        raw_id = str(row[disease_id_col]).strip()
        norm_id = normalize_id(raw_id)
        name = str(row[disease_name_col]).strip()
        disease_map[raw_id] = name
        disease_map[norm_id] = name
        disease_map[f"D{norm_id}"] = name
        disease_desc_map[raw_id] = name
        disease_desc_map[norm_id] = name
        disease_desc_map[f"D{norm_id}"] = name
    return drug_map, disease_map, disease_desc_map


def load_ordered_ids(mat) -> Tuple[List[str], List[str]]:
    ordered_drug_ids = [str(item[0][0]).strip() for item in mat["Wrname"]]
    ordered_disease_ids = [str(item[0][0]).strip() for item in mat["Wdname"]]
    return ordered_drug_ids, ordered_disease_ids


def find_target_disease(
    query: str,
    ordered_disease_ids: List[str],
    disease_map: Dict[str, str],
) -> Tuple[int, str, str]:
    q = query.lower().strip()
    for idx, disease_id in enumerate(ordered_disease_ids):
        norm_id = normalize_id(disease_id)
        disease_text = disease_map.get(disease_id, disease_map.get(norm_id, ""))
        haystacks = {disease_id.lower(), norm_id.lower(), disease_text.lower()}
        if any(q in item for item in haystacks):
            return idx, disease_id, disease_text or disease_id
    return -1, "", ""


def drugbank_url(drug_id: str) -> str:
    return f"https://go.drugbank.com/drugs/{drug_id}"


def clinical_trials_url(drug_id: str, disease_id: str) -> str:
    query = f"{drug_id}%20{normalize_id(disease_id)}"
    return f"https://clinicaltrials.gov/search?term={query}"


def run_dataset(dataset_name: str):
    data_root = os.path.join(PROJECT_ROOT, "data", dataset_name)
    result_dir = os.path.join(PROJECT_ROOT, "results", dataset_name, "case_study")
    os.makedirs(result_dir, exist_ok=True)

    mat_path = os.path.join(data_root, f"{dataset_name}.mat")
    feat_dir = os.path.join(data_root, "features")
    proc_dir = os.path.join(data_root, "processed")
    drug_csv = os.path.join(data_root, "drug_desc.csv")
    disease_csv = os.path.join(data_root, "disease_desc.csv")
    weight_path = os.environ.get("MODEL_WEIGHT_PATH", os.path.join(proc_dir, "best_model.pth"))
    dataset_pt = os.path.join(proc_dir, "dataset.pt")

    print("\n" + "=" * 70)
    print(f"Case Study: {dataset_name}")
    print("=" * 70)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing model weight: {weight_path}")
    if not os.path.exists(dataset_pt):
        raise FileNotFoundError(f"Missing single-split graph dataset.pt: {dataset_pt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat = sio.loadmat(mat_path)
    drug_sim = mat["drug"].astype(np.float32)
    disease_sim = mat["disease"].astype(np.float32)
    adj = mat["didr"]
    if adj.shape == (disease_sim.shape[0], drug_sim.shape[0]):
        adj = adj.T
    adj = adj.astype(np.float32)

    ordered_drug_ids, ordered_disease_ids = load_ordered_ids(mat)
    drug_map, disease_map, _ = load_name_maps(drug_csv, disease_csv)

    graph_data = torch.load(dataset_pt, weights_only=False)
    adj_dd = torch.from_numpy(graph_data["adj_dd"]).float().to(device)
    adj_ss = torch.from_numpy(graph_data["adj_ss"]).float().to(device)
    adj_ds = torch.from_numpy(graph_data["adj_ds"]).float().to(device)
    d_feat = torch.from_numpy(np.load(os.path.join(feat_dir, "drug_features.npy"))).float().to(device)
    s_feat = torch.from_numpy(np.load(os.path.join(feat_dir, "disease_features.npy"))).float().to(device)

    model = DReKGNN_PyTorch(
        input_dim=d_feat.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        edge_drop=0.0,
        num_drugs=adj.shape[0],
        num_diseases=adj.shape[1],
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        agg_drug, agg_disease = model(d_feat, s_feat, adj_dd, adj_ss, adj_ds)

    rows = []
    for target in TARGET_DISEASES:
        disease_idx, disease_id, disease_name = find_target_disease(target, ordered_disease_ids, disease_map)
        if disease_idx < 0:
            print(f"[WARN] Disease target not found: {target}")
            continue

        unknown_drug_indices = np.where(adj[:, disease_idx] == 0)[0]
        test_x = torch.tensor(
            [[drug_idx, disease_idx] for drug_idx in unknown_drug_indices],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits = model.predict(agg_drug, agg_disease, test_x)
            probs = torch.sigmoid(logits).cpu().numpy()

        order = np.argsort(probs)[::-1][:TOP_K]
        print(f"\nTarget disease {target} -> {disease_id} | candidates={len(unknown_drug_indices)}")
        print(f"{'Rank':<5} {'Drug_ID':<10} {'Score':<10} Drug description")
        for rank, rel_idx in enumerate(order, start=1):
            drug_idx = int(unknown_drug_indices[rel_idx])
            drug_id = ordered_drug_ids[drug_idx]
            drug_name = drug_map.get(drug_id, drug_id)
            score = float(probs[rel_idx])
            short_name = drug_name[:90].replace("\n", " ")
            print(f"{rank:<5} {drug_id:<10} {score:<10.6f} {short_name}")
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Disease_Query": target,
                    "Disease_ID": disease_id,
                    "Disease_OMIM": normalize_id(disease_id),
                    "Disease_Description": disease_name,
                    "Rank": rank,
                    "Drug_Index": drug_idx,
                    "Drug_ID": drug_id,
                    "Drug_Description": drug_name,
                    "Score": score,
                    "DrugBank_URL": drugbank_url(drug_id),
                    "ClinicalTrials_Query_URL": clinical_trials_url(drug_id, disease_id),
                }
            )

    if rows:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            result_dir,
            f"{dataset_name}_case_study_top{TOP_K}_{timestamp}.csv",
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nSaved case study results to:\n  {out_path}")
        return out_path
    print("No case study rows generated.")
    return None


def main():
    print(f"Datasets: {DATASETS}")
    print(f"Targets: {TARGET_DISEASES}")
    print(f"Top-K: {TOP_K}")
    outputs = []
    for dataset_name in DATASETS:
        outputs.append(run_dataset(dataset_name))
    print("\nAll outputs:")
    for path in outputs:
        if path:
            print(path)


if __name__ == "__main__":
    main()
