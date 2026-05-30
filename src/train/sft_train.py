import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 为了能顺利导入 src.utils.tools，将项目根目录临时加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

def prepare_dataset(tokenizer, data_path, max_length=128):
    """加载并处理数据集"""
    # 加载 JSONL 数据
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # 🔴 CPU 调试特供：为了快速验证，我们只取前 50 条数据
    print("⚠️ 检测到测试模式：截取前 50 条数据进行快速验证...")
    dataset = dataset.select(range(50))
    
    def tokenize_function(example):
        messages = [
            {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        # 使用 Qwen 的对话模板
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 转换为 Token IDs
        model_inputs = tokenizer(text, max_length=max_length, truncation=True, padding="max_length")
        
        # Causal LM 任务中，labels 与 input_ids 一致，模型内部会自动做 shift 操作
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    return tokenized_dataset

def main():
    root_dir = get_project_root()
    model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen1.5-0.5B-Chat") 
    data_path = os.path.join(root_dir, "data", "processed", "tang_qwen_sft.jsonl")
    output_dir = os.path.join(root_dir, "models", "checkpoints")

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # Qwen 的 pad_token 默认可能为空，需要设置一下以支持 batch 处理
    tokenizer.pad_token_id = tokenizer.eod_id

    print("2. 正在准备数据集...")
    train_dataset = prepare_dataset(tokenizer, data_path)

    print("3. 正在加载基础模型 (CPU 模式)...")
    # 🔴 CPU 调试特供：指定 device_map="cpu"，不使用任何量化或半精度
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # 禁用缓存以节省内存（训练时必须）
    model.config.use_cache = False

    print("4. 配置 LoRA 适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Qwen 全连接层
        inference_mode=False, # 训练模式
        r=8,                  # 秩，越小参数越少
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # 打印一下看我们冻结了多少参数

    print("5. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        use_cpu=True,                   # 🔴 强制使用 CPU
        per_device_train_batch_size=1,  # CPU 内存有限，Batch Size 设为 1
        gradient_accumulation_steps=4,  # 模拟更大的 Batch Size (1*4=4)
        max_steps=10,                   # 🔴 调试特供：只跑 10 步就结束，验证流程通畅
        logging_steps=1,
        save_steps=10,
        learning_rate=3e-4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("🚀 开始训练...")
    trainer.train()
    
    print(f"✅ 验证成功！LoRA 权重已保存在: {output_dir}")

if __name__ == "__main__":
    main()