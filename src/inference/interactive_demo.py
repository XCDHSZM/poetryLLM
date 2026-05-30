import os
import sys
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 临时将项目根目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

def get_latest_checkpoint(checkpoint_dir):
    """自动获取最新的 checkpoint 文件夹"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"❌ 在 {checkpoint_dir} 下没有找到任何 checkpoint 文件夹！")
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

# ==========================================
# 核心生成函数：只负责写诗，不负责加载模型
# ==========================================
def generate_poem(instruction, model, tokenizer):
    """
    接受单句诗词，调用已挂载 LoRA 的大模型生成续写
    """
    messages = [
        {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
        {"role": "user", "content": instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )
    
    # 将输入推送到 GPU
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 禁用梯度计算以加速推理
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=50,
            temperature=0.7,         
            top_p=0.9,               
            repetition_penalty=1.1,  
        )

    # 截取模型新生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# ==========================================
# 主程序：加载环境并开启对话循环
# ==========================================
def main():
    root_dir = get_project_root()
    base_model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen2.5-3B-Instruct")
    checkpoint_base_dir = os.path.join(root_dir, "models", "checkpoints")
    
    try:
        lora_dir = get_latest_checkpoint(checkpoint_base_dir)
    except Exception as e:
        print(e)
        return

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. 正在加载基座大模型 (大约需要1-2分钟)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"3. 正在挂载 LoRA 灵魂 ({os.path.basename(lora_dir)})...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval() 

    print("\n" + "="*50)
    print("🎉 AI 诗人已就位！(输入 'quit' 或 'exit' 退出测试)")
    print("="*50 + "\n")

    # 4. 无限循环，随性测试
    # while True:
    #     try:
    #         # 接收终端输入
    #         user_input = input("👉 请输入古诗首句 (例如: 孤舟蓑笠翁，): ").strip()
            
    #         # 退出指令检测
    #         if user_input.lower() in ['quit', 'exit']:
    #             print("👋 期待与您下次品茗论诗！")
    #             break
                
    #         if not user_input:
    #             continue

    #         print("📝 模型构思中...\n")
            
    #         # 调用生成函数
    #         result = generate_poem(user_input, model, tokenizer)
            
    #         print(f"【AI 续写】\n{result}\n")
    #         print("-" * 50)
            
    #     except KeyboardInterrupt:
    #         # 捕获 Ctrl+C，优雅退出
    #         print("\n👋 已强制退出。")
    #         break
    print("测试古诗“孤舟蓑笠翁”")
    user_input = "孤舟蓑笠翁"
    print("📝 模型构思中...\n")
    result = generate_poem(user_input, model, tokenizer)
    print(f"【AI 续写】\n{result}\n")

if __name__ == "__main__":
    main()