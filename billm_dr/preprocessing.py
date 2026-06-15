import os
import pandas as pd
import scipy.io as sio
import re


def run_preprocessing(mat_path, drug_csv, disease_csv, output_dir):
    print("\n--- 开始数据预处理 ---")
    os.makedirs(output_dir, exist_ok=True)

    def clean_and_mask(text):
        if not isinstance(text, str) or text.lower() == 'nan':
            return "No description available."
        text = text.lower()
        mask_patterns = [r"used to treat", r"indicated for", r"treatment of", r"effective against", r"approved for"]
        for pattern in mask_patterns:
            text = re.sub(pattern, "[RELATION_MASKED]", text)
        return text

    # 读取索引顺序
    mat = sio.loadmat(mat_path)
    ordered_drug_ids = [str(i[0][0]).strip() for i in mat['Wrname']]
    ordered_disease_ids = [str(i[0][0]).strip() for i in mat['Wdname']]

    drug_df = pd.read_csv(drug_csv)
    disease_df = pd.read_csv(disease_csv)

    drug_map = dict(zip(drug_df['db_id'].astype(str).str.strip(), drug_df['Description']))
    disease_map = dict(zip(disease_df['omim_id'].astype(str).str.strip(), disease_df['Description']))

    # 对齐文本
    final_drugs = [clean_and_mask(drug_map.get(d_id, "No description available.")) for d_id in ordered_drug_ids]
    final_diseases = [clean_and_mask(disease_map.get(dis_id, "No description available.")) for dis_id in
                      ordered_disease_ids]

    pd.DataFrame({'text': final_drugs}).to_csv(os.path.join(output_dir, "aligned_drugs.csv"), index=False)
    pd.DataFrame({'text': final_diseases}).to_csv(os.path.join(output_dir, "aligned_diseases.csv"), index=False)
    print(f"✅ 预处理完成，对齐文本已保存至: {output_dir}")