# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# import os
#
#
# def run_feature_extraction(model_path, input_dir, save_path):
#     print("\n--- 开始大模型特征提取 ---")
#     os.makedirs(save_path, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     model = AutoModel.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
#     model.eval()
#
#     def process(csv_name, npy_name):
#         texts = pd.read_csv(os.path.join(input_dir, csv_name))['text'].tolist()
#         all_embs = []
#         batch_size = 2
#         for i in tqdm(range(0, len(texts), batch_size), desc=f"处理 {npy_name}"):
#             batch = texts[i:i + batch_size]
#             inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#             mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
#             sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
#             mean_emb = sum_emb / torch.clamp(mask.sum(1), min=1e-9)
#             all_embs.append(mean_emb.cpu().to(torch.float32).numpy())
#
#         np.save(os.path.join(save_path, npy_name), np.vstack(all_embs))
#
#     process("aligned_drugs.csv", "drug_features.npy")
#     process("aligned_diseases.csv", "disease_features.npy")
#     print(f"✅ 特征提取完成，已保存至: {save_path}")


import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import os


def run_feature_extraction(model_path, input_dir, save_path):
    model_name = os.path.basename(model_path.strip('/'))
    print(f"\n--- 开始大模型特征提取 ({model_name}) ---")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 兼容性极强的 Tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        # 兼容 Llama-3 / Qwen 等没有默认 pad_token 的情况
        tokenizer.pad_token = tokenizer.eos_token

    print("正在将模型权重加载至 GPU，请稍候...")
    print("正在将模型权重加载至 GPU，请稍候...")
    # 2. 自动架构识别加载
    try:
        # 尝试作为基础 Encoder 加载
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 🚀 这里修复为 torch_dtype
            device_map="auto",
            trust_remote_code=True
        )
    except:
        # 如果失败，说明它是纯生成式 Decoder 模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 🚀 这里修复为 torch_dtype
            device_map="auto",
            trust_remote_code=True
        )
        # 剥离 LM Head 预测头，只取底层特征抽取器
        if hasattr(model, 'model'):
            model = model.model


    model.eval()

    def process(csv_name, npy_name):
        texts = pd.read_csv(os.path.join(input_dir, csv_name))['text'].tolist()
        all_embs = []
        batch_size = 2  # 如果你的显存很大(如 24G)，可以调到 4 或 8 加速提取

        for i in tqdm(range(0, len(texts), batch_size), desc=f"处理 {npy_name}"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                # 强制输出所有隐藏层状态
                outputs = model(**inputs, output_hidden_states=True)

            # 3. 动态获取最后一层隐藏层输出
            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_state = outputs.hidden_states[-1]
            else:
                hidden_state = outputs[0]

            # 4. Mean Pooling 均值池化 (去除 Padding 部分的噪音)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
            sum_emb = torch.sum(hidden_state * mask, 1)
            mean_emb = sum_emb / torch.clamp(mask.sum(1), min=1e-9)

            all_embs.append(mean_emb.cpu().to(torch.float32).numpy())

        np.save(os.path.join(save_path, npy_name), np.vstack(all_embs))

    process("aligned_drugs.csv", "drug_features.npy")
    process("aligned_diseases.csv", "disease_features.npy")
    print(f"✅ 特征提取完成，已保存至: {save_path}")