import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 将项目根目录临时加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

def main():
    root_dir = get_project_root()
    # 基础模型路径 (Qwen 本尊)
    base_model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen1.5-0.5B-Chat")
    # LoRA 权重路径 (刚刚训练出来的小本本)
    # 注意：这里的 checkpoint-10 是因为我们上面代码设置了 max_steps=10。
    # 如果你的文件夹名字不一样，请修改这里！
    lora_dir = os.path.join(root_dir, "models", "checkpoints", "checkpoint-10")

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)

    print("2. 正在加载基础模型 (CPU 模式)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    print("3. 正在将 LoRA 权重挂载到基础模型上...")
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"找不到 LoRA 权重文件夹：{lora_dir}")
    
    # 核心步骤：把基础模型和 LoRA 权重合并在一起
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval() # 切换到评估/推理模式

    print("\n" + "="*50)
    print("🤖 模型加载完毕，开始写诗！")
    print("="*50 + "\n")

    # 4. 准备测试题目（呼应你指导书中的要求）
    test_instruction = "请续写这首唐诗：湖光秋月两相和，"
    
    # 严格按照训练时的对话模板进行组装
    messages = [
        {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
        {"role": "user", "content": test_instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # 推理时设为 True，会加上 <|im_start|>assistant 引导模型开始回答
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    print(f"【输入提示】 {test_instruction}\n")
    print("【模型生成中，请稍候...】")

    # 5. 执行生成
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # 👈 新增这一行，把 mask 传给模型
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # 6. 解码输出
    # 截取掉输入部分，只保留模型新生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"【AI 续写】 {response}\n")

if __name__ == "__main__":
    main()