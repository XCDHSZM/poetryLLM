import numpy as np
import json
import os
def get_project_root():
    """
    动态获取项目根目录
    假设当前脚本位于 项目根目录/src/data_process/make_dataset.py
    """
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 向上推三级目录：make_dataset.py -> data_process -> src -> 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    return project_root

def convert_npz_to_jsonl():
    # 1. 动态构建输入和输出的绝对路径
    root_dir = get_project_root()
    input_path = os.path.join(root_dir, "data", "raw", "tang.npz")
    output_path = os.path.join(root_dir, "data", "processed", "tang_qwen_sft.jsonl")
    
    # 确保输出目录存在，如果不存在则自动创建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"正在从以下路径加载原始数据：\n{input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到 tang.npz，请检查是否已将文件放入 data/raw/ 目录！")

    # 2. 加载 npz 文件
    datas = np.load(input_path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    
    success_count = 0
    print("开始清洗与转换...")
    
    # 3. 解析与写入
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data:
            # 逆向解码：数字索引转汉字
            poem_chars = [ix2word[idx] for idx in row]
            poem_text = "".join(poem_chars)
            
            # 清洗特殊标记
            poem_text = poem_text.replace("</s>", "").replace("<START>", "").replace("<EOP>", "").strip()
            
            # 寻找切分点（以第一个常见句读为界）
            split_idx = -1
            for i, char in enumerate(poem_text):
                if char in ['。', '，', '？', '！']:
                    split_idx = i + 1
                    break
            
            # 确保切分有效
            if split_idx != -1 and split_idx < len(poem_text):
                first_sentence = poem_text[:split_idx]
                rest_of_poem = poem_text[split_idx:]
                
                # 组装 Instruction Tuning (SFT) 格式
                json_record = {
                    "instruction": f"请续写这首唐诗：{first_sentence}",
                    "input": "",
                    "output": rest_of_poem
                }
                
                f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                success_count += 1
                
    print(f"✅ 数据清洗与转换完成！共生成 {success_count} 条有效训练数据。")
    print(f"✅ 微调数据已保存至：\n{output_path}")
if __name__ == "__main__":
    convert_npz_to_jsonl()