# ==========================================
# 🚨 关键修改：导入我们刚刚新建的带 '1' 的新脚本！
# ==========================================
from billm_dr import preprocessing1 as preprocessing
from billm_dr import feature_extraction1 as feature_extraction
from billm_dr import data_split1 as data_split
from billm_dr import train_eval1
import os

ROOT = os.environ.get("DATA_ROOT", "/home/zqgaopengfei/project/zym/data") + os.sep
# 🚀 换成你的新数据集文件夹名称
DATASET_NAME = os.environ.get("DATASET_NAME", "Rdataset")

CONFIG = {
    # ==========================================
    # 📁 数据集路径配置 (针对 CSV 格式新数据集)
    # ==========================================
    "drug_dis_csv": ROOT + f"{DATASET_NAME}/drug_dis.csv",  # 药物-疾病关联矩阵/边表
    "drug_sim_csv": ROOT + f"{DATASET_NAME}/drug_sim.csv",  # 药物相似度
    "dis_sim_csv": ROOT + f"{DATASET_NAME}/dis_sim.csv",  # 疾病相似度
    "drug_desc_csv": ROOT + f"{DATASET_NAME}/drug_desc.csv",  # 药物描述文本
    "disease_desc_csv": ROOT + f"{DATASET_NAME}/disease_desc.csv",  # 疾病描述文本

    "model_path": os.environ.get("MODEL_PATH", os.path.join(ROOT, "LLM", "Qwen", "Qwen3-8B")),
    "proc_dir": ROOT + f"{DATASET_NAME}/processed/",
    "feat_dir": ROOT + f"{DATASET_NAME}/features/",

    # ==========================================
    # 🎯 模型与训练可调超参数集中管理区
    # ==========================================
    "run_mode": "10-fold",
    "seed": 42,
    "hidden_dim": 512,
    "dropout": 0.7,
    "edge_drop": 0.5,
    "batch_size": 512,
    "epochs": 2000,
    "patience": 15,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "pos_weight": 10.0,
    "margin": 1.0,
    "bpr_weight": 0.5,
    "top_k": 6000,
}
print(CONFIG)


def main():
    # 确保输出目录存在
    os.makedirs(CONFIG['proc_dir'], exist_ok=True)
    os.makedirs(CONFIG['feat_dir'], exist_ok=True)

    # 1. 预处理 (对齐文本，可能需要转换 CSV 到图结构)
    # 因为上面 import 的是 preprocessing1，这里实际调用的就是新版 CSV 预处理
    preprocessing.run_preprocessing(
        CONFIG['drug_dis_csv'],
        CONFIG['drug_desc_csv'],
        CONFIG['disease_desc_csv'],
        CONFIG['proc_dir']
    )

    # 2. 特征提取 (读取上一步处理好的对齐 json/csv 文本)
    if not os.path.exists(CONFIG['feat_dir'] + "drug_features.npy"):
        feature_extraction.run_feature_extraction(CONFIG['model_path'], CONFIG['proc_dir'], CONFIG['feat_dir'])
    else:
        print("\n检测到特征文件已存在，跳过大模型提取步骤。")

    # 3 & 4. 路由逻辑
    if CONFIG['run_mode'] == "single":
        print("\n>>> 模式检测：当前为 [单次验证模式]")
        data_split.run_data_split(CONFIG)
        train_eval1.run_single_eval(CONFIG)

    elif CONFIG['run_mode'] == "10-fold":
        print("\n>>> 模式检测：当前为 [10折交叉验证模式]")
        train_eval1.run_10fold_eval(CONFIG)

    else:
        raise ValueError(f"未知的 run_mode: {CONFIG['run_mode']}，请选择 'single' 或 '10-fold'")


if __name__ == "__main__":
    main()