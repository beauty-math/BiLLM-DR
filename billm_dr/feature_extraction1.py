import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import os


def run_feature_extraction(model_path, input_dir, save_path):
    model_name = os.path.basename(model_path.strip('/'))
    print(f"\n--- 开始大模型特征提取 ({model_name}) (CSV 版本) ---")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 兼容性极强的 Tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在将模型权重加载至 GPU，请稍候...")

    # 2. 自动架构识别加载
    try:
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        if hasattr(model, 'model'):
            model = model.model

    model.eval()

    def process(csv_name, npy_name):
        texts = pd.read_csv(os.path.join(input_dir, csv_name))['text'].tolist()
        all_embs = []
        batch_size = 2

        for i in tqdm(range(0, len(texts), batch_size), desc=f"处理 {npy_name}"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_state = outputs.hidden_states[-1]
            else:
                hidden_state = outputs[0]

            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
            sum_emb = torch.sum(hidden_state * mask, 1)
            mean_emb = sum_emb / torch.clamp(mask.sum(1), min=1e-9)

            all_embs.append(mean_emb.cpu().to(torch.float32).numpy())

        np.save(os.path.join(save_path, npy_name), np.vstack(all_embs))

    process("aligned_drugs.csv", "drug_features.npy")
    process("aligned_diseases.csv", "disease_features.npy")
    print(f"✅ 特征提取完成，已保存至: {save_path}")