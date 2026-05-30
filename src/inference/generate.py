import os
import sys
import glob
import torch
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
    
    # 按照 checkpoint 后面的数字大小进行排序，取最大的（即最新的）
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    root_dir = get_project_root()
    
    # 🔴 1. 更新基础模型路径为 3B 模型
    base_model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen2.5-3B-Instruct")
    
    # 🔴 2. 动态获取最新的 LoRA 权重路径
    checkpoint_base_dir = os.path.join(root_dir, "models", "checkpoints")
    lora_dir = get_latest_checkpoint(checkpoint_base_dir)

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. 正在加载基础模型 (GPU 模式 & FP16 半精度)...")
    # 🔴 3. 将 device_map 改为 auto（使用 GPU），并将精度改为 float16 以匹配训练状态
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"3. 正在将最新的 LoRA 权重挂载到基础模型上...\n   挂载路径: {lora_dir}")
    # 核心步骤：把基础模型和 LoRA 权重合并在一起
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval() # 切换到评估/推理模式

    print("\n" + "="*50)
    print("🤖 诗词大模型加载完毕，请出题！")
    print("="*50 + "\n")

    # 4. 准备测试题目
    test_instruction = "请续写这首唐诗：湖光秋月两相和，"
    
    messages = [
        {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
        {"role": "user", "content": test_instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )
    
    # 🔴 4. 将输入数据也推送到模型所在的设备（GPU）上
    model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    print(f"【输入提示】 {test_instruction}\n")
    print("【模型生成中，请稍候...】")

    # 5. 执行生成
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=50,
            temperature=0.7,         
            top_p=0.9,               
            repetition_penalty=1.1,  
        )

    # 6. 解码输出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"【AI 续写】 {response}\n")

if __name__ == "__main__":
    main()