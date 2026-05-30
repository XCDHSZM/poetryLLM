import os
import sys
import glob
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 将项目根目录临时加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

def get_latest_checkpoint(checkpoint_dir):
    """自动获取最新的 checkpoint 文件夹"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"❌ 在 {checkpoint_dir} 下没有找到任何 checkpoint 文件夹！请先进行训练。")
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    root_dir = get_project_root()
    
    # 路径配置
    base_model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen2.5-3B-Instruct")
    checkpoint_base_dir = os.path.join(root_dir, "models", "checkpoints")
    lora_dir = get_latest_checkpoint(checkpoint_base_dir)
    
    # 数据集路径配置
    val_data_path = os.path.join(root_dir, "data", "processed", "tang_val.jsonl")
    output_results_path = os.path.join(root_dir, "data", "processed", "eval_results.jsonl")

    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"❌ 找不到验证集文件：{val_data_path}")

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. 正在加载基础模型 (GPU 模式 & FP16 半精度)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"3. 正在将最新的 LoRA 权重挂载到基础模型上...\n   挂载路径: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    print("\n" + "="*50)
    print("🤖 模型加载完毕，开始批量生成评估数据！")
    print("="*50 + "\n")

    # 读取验证集数据
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_records = [json.loads(line.strip()) for line in f]

    print(f"📦 共读取到 {len(val_records)} 条验证数据，准备开始推理...")

    # 打开输出文件准备写入
    with open(output_results_path, 'w', encoding='utf-8') as outfile:
        # 使用 tqdm 包装循环，显示进度条
        for record in tqdm(val_records, desc="批量推理进度"):
            instruction = record["instruction"]
            target = record["output"] # 原数据集里的 output 就是 Ground Truth
            
            messages = [
                {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
                {"role": "user", "content": instruction}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True 
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

            # 执行生成
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=50,
                    temperature=0.7,         
                    top_p=0.9,               
                    repetition_penalty=1.1,  
                )

            # 解码输出
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 组装用于评估的 JSON 格式
            result_record = {
                "instruction": instruction,
                "target": target,
                "generated": response
            }
            
            # 写入文件
            outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print(f"\n✅ 批量推理完成！结果已保存至：{output_results_path}")
    print("💡 接下来，你可以运行 src/evaluate/run_eval.py 来生成最终的量化评估报告了！")

if __name__ == "__main__":
    main()