from billm_dr import preprocessing
from billm_dr import feature_extraction
from billm_dr import data_split
from billm_dr import train_eval
import os


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


ROOT = os.environ.get("DATA_ROOT", "/home/zqgaopengfei/project/zym/data")
DATASET_NAME = os.environ.get("DATASET_NAME", "Cdataset")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "LLM", "Qwen", "Qwen3-8B"))

CONFIG = {
    "mat_path": os.path.join(ROOT, DATASET_NAME, f"{DATASET_NAME}.mat"),
    "drug_csv": os.path.join(ROOT, DATASET_NAME, "drug_desc.csv"),
    "disease_csv": os.path.join(ROOT, DATASET_NAME, "disease_desc.csv"),
    "model_path": MODEL_PATH,
    "proc_dir": os.path.join(ROOT, DATASET_NAME, "processed") + os.sep,
    "feat_dir": os.path.join(ROOT, DATASET_NAME, "features") + os.sep,
    "run_mode": os.environ.get("RUN_MODE", "10-fold"),
    "seed": _env_int("SEED", 42),
    "hidden_dim": _env_int("HIDDEN_DIM", 512),
    "dropout": _env_float("DROPOUT", 0.5),
    "edge_drop": _env_float("EDGE_DROP", 0.25),
    "batch_size": _env_int("BATCH_SIZE", 512),
    "epochs": _env_int("EPOCHS", 2000),
    "patience": _env_int("PATIENCE", 15),
    "lr": _env_float("LR", 1e-4),
    "weight_decay": _env_float("WEIGHT_DECAY", 5e-4),
    "pos_weight": _env_float("POS_WEIGHT", 12.0),
    "neg_ratio": _env_int("NEG_RATIO", 90),
    "knn_k": _env_int("KNN_K", 15),
    "margin": _env_float("MARGIN", 1.0),
    "bpr_weight": _env_float("BPR_WEIGHT", 0.3),
    "hard_bpr": _env_bool("HARD_BPR", False),
    "top_k": _env_int("TOP_K", 6000),
}
print(CONFIG)


def main():
    preprocessing.run_preprocessing(CONFIG['mat_path'], CONFIG['drug_csv'], CONFIG['disease_csv'], CONFIG['proc_dir'])

    if not os.path.exists(os.path.join(CONFIG['feat_dir'], "drug_features.npy")):
        feature_extraction.run_feature_extraction(CONFIG['model_path'], CONFIG['proc_dir'], CONFIG['feat_dir'])
    else:
        print("\n?????????????????????")

    if CONFIG['run_mode'] == "single":
        print("\n>>> ???????? [??????]")
        data_split.run_data_split(CONFIG['mat_path'], CONFIG['proc_dir'])
        train_eval.run_single_eval(CONFIG)
    elif CONFIG['run_mode'] == "10-fold":
        print("\n>>> ???????? [10???????]")
        train_eval.run_10fold_eval(CONFIG)
    else:
        raise ValueError(f"??? run_mode: {CONFIG['run_mode']}???? 'single' ? '10-fold'")


if __name__ == "__main__":
    main()
