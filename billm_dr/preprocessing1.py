import os
import re
import pandas as pd
import numpy as np


def load_csv_matrix(filepath):
    """自适应读取 CSV 矩阵文件"""
    try:
        df = pd.read_csv(filepath, header=None)
        if df.iloc[0].apply(lambda x: isinstance(x, str)).any():
            df = pd.read_csv(filepath, index_col=0)
        return df.values.astype(np.float32)
    except Exception as e:
        raise ValueError(f"读取 CSV 文件 {filepath} 失败: {e}")


def run_preprocessing(drug_dis_csv, drug_desc_csv, disease_desc_csv, output_dir):
    print("\n--- 开始数据预处理 (CSV 新版本 - 修复维度锁定版) ---")
    os.makedirs(output_dir, exist_ok=True)

    def clean_and_mask(text):
        if not isinstance(text, str) or text.lower() == 'nan':
            return "No description available."
        text = text.lower()
        mask_patterns = [r"used to treat", r"indicated for", r"treatment of", r"effective against", r"approved for"]
        for pattern in mask_patterns:
            text = re.sub(pattern, "[RELATION_MASKED]", text)
        return text

    # 🚀 绝对精准的维度获取：直接去找同目录下的 similarity 矩阵获取维度
    dataset_dir = os.path.dirname(drug_dis_csv)
    drug_sim_csv = os.path.join(dataset_dir, 'drug_sim.csv')
    dis_sim_csv = os.path.join(dataset_dir, 'dis_sim.csv')

    try:
        num_drugs = load_csv_matrix(drug_sim_csv).shape[0]
        num_diseases = load_csv_matrix(dis_sim_csv).shape[0]
    except Exception as e:
        print(f"⚠️ 无法读取相似度矩阵获取维度，回退到使用关联矩阵... {e}")
        adj = load_csv_matrix(drug_dis_csv)
        num_drugs = adj.shape[0]
        num_diseases = adj.shape[1]

    print(f"✅ 锁定真实图维度: 药物 {num_drugs} 个, 疾病 {num_diseases} 个")

    # 读取文本描述并清洗
    def align_descriptions(csv_path, expected_num, entity_type):
        df = pd.read_csv(csv_path)

        # 尝试寻找描述列 (优先找 Description/text 列，否则取最后一列)
        col_name = None
        for col in df.columns:
            if col.lower() in ['description', 'text', 'desc']:
                col_name = col
                break
        if not col_name:
            col_name = df.columns[-1]

        raw_texts = df[col_name].tolist()

        # 对齐数量 (多了截断，少了用占位符补齐)
        actual_num = len(raw_texts)
        if actual_num != expected_num:
            print(
                f"⚠️ 警告: {entity_type} 描述数量 ({actual_num}) 与矩阵维度 ({expected_num}) 不匹配，已自动截断或填充。")
            if actual_num < expected_num:
                raw_texts.extend(["No description available."] * (expected_num - actual_num))
            else:
                raw_texts = raw_texts[:expected_num]

        return [clean_and_mask(str(text)) for text in raw_texts]

    final_drugs = align_descriptions(drug_desc_csv, num_drugs, "Drug")
    final_diseases = align_descriptions(disease_desc_csv, num_diseases, "Disease")

    # 落盘保存
    pd.DataFrame({'text': final_drugs}).to_csv(os.path.join(output_dir, "aligned_drugs.csv"), index=False)
    pd.DataFrame({'text': final_diseases}).to_csv(os.path.join(output_dir, "aligned_diseases.csv"), index=False)
    print(f"✅ 预处理完成，对齐文本已保存至: {output_dir}")