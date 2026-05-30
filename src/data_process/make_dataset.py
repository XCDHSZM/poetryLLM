import numpy as np
import json
import os
import random
from src.utils.tools import get_project_root

def convert_npz_to_jsonl():
    root_dir = get_project_root()
    input_path = os.path.join(root_dir, "data", "raw", "tang.npz")
    
    # 定义训练集和验证集的输出路径
    train_output_path = os.path.join(root_dir, "data", "processed", "tang_train.jsonl")
    val_output_path = os.path.join(root_dir, "data", "processed", "tang_val.jsonl")
    
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)

    print(f"正在从以下路径加载原始数据：\n{input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到 tang.npz，请检查文件路径！")

    datas = np.load(input_path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    
    print("开始清洗与转换...")
    
    all_valid_records = []
    
    # 1. 提取并清洗所有数据
    for row in data:
        poem_chars = [ix2word[idx] for idx in row]
        poem_text = "".join(poem_chars)
        poem_text = poem_text.replace("</s>", "").replace("<START>", "").replace("<EOP>", "").strip()
        
        split_idx = -1
        for i, char in enumerate(poem_text):
            if char in ['。', '，', '？', '！']:
                split_idx = i + 1
                break
        
        if split_idx != -1 and split_idx < len(poem_text):
            first_sentence = poem_text[:split_idx]
            rest_of_poem = poem_text[split_idx:]
            
            json_record = {
                "instruction": f"请续写这首唐诗：{first_sentence}",
                "input": "",
                "output": rest_of_poem
            }
            all_valid_records.append(json_record)
            
    print(f"共提取出 {len(all_valid_records)} 条有效诗词数据。")
    
    # 2. 随机打乱数据 (设定随机种子保证每次打乱结果一致，便于复现)
    random.seed(42)
    random.shuffle(all_valid_records)
    
    # 3. 划分数据集：预留 1000 条作为测试/验证集，剩下全部用于训练
    val_size = 1000
    val_records = all_valid_records[:val_size]
    train_records = all_valid_records[val_size:]
    
    # 4. 分别写入文件
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    with open(val_output_path, 'w', encoding='utf-8') as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("✅ 数据清洗与划分完成！")
    print(f" - 训练集 (Train): {len(train_records)} 条 -> {train_output_path}")
    print(f" - 验证集 (Validation/Test): {len(val_records)} 条 -> {val_output_path}")

if __name__ == "__main__":
    convert_npz_to_jsonl()