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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

def prepare_dataset(tokenizer, data_path, max_length=128):
    """加载并处理完整训练集"""
    # 加载 JSONL 数据
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # 🔴 封印解除：移除了 dataset.select(range(50))，现在使用全量数据！
    print(f"📦 成功加载训练集，共 {len(dataset)} 条数据。")
    
    def tokenize_function(example):
        messages = [
            {"role": "system", "content": "你是一个精通中国古典诗词的AI诗人。"},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer(text, max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    return tokenized_dataset

def main():
    root_dir = get_project_root()
    
    # 🔴 路径更新：指向你刚下载的 3B 大模型和昨天切分好的全量训练集
    model_dir = os.path.join(root_dir, "models", "pretrained", "qwen", "Qwen1___5-0___5B-Chat") 
    data_path = os.path.join(root_dir, "data", "processed", "tang_train.jsonl")
    output_dir = os.path.join(root_dir, "models", "checkpoints")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ 找不到模型文件夹：{model_dir}。请检查下载脚本是否运行成功。")

    print("1. 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 修复之前的 pad_token 报错
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. 正在准备数据集...")
    train_dataset = prepare_dataset(tokenizer, data_path)

    print("3. 正在加载基础模型 (双卡并行 & FP16 半精度加速)...")
    # 🔴 核心显卡配置：device_map="auto" 会自动把模型切分到两张 T4 显卡上
    # 🔴 torch_dtype=torch.float16：T4 显卡对 FP16 有硬件级加速，能省一半显存并大幅提速
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    model.config.use_cache = False

    print("4. 配置 LoRA 适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    print("5. 配置工业级分布式训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        fp16=True,                      # 🔴 开启混合精度训练 (极其关键)
        per_device_train_batch_size=4,  # 每张卡塞 4 条数据 (双卡实际 Batch Size = 8)
        gradient_accumulation_steps=2,  # 梯度累加，等效总 Batch Size = 16
        num_train_epochs=3,             # 🔴 完整遍历所有唐诗 3 遍
        learning_rate=3e-4,
        logging_steps=50,               # 每 50 步打印一次 Loss 观察是否收敛
        save_strategy="epoch",          # 每个 epoch 结束保存一次模型
        save_total_limit=2,             # 最多只保留最近的 2 个权重，防止 Kaggle 硬盘撑爆
        remove_unused_columns=False,
        report_to="none"                # 禁用 wandb 等外部日志上报
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("🚀 开始在 GPU 集群上全量训练...")
    trainer.train()
    
    print(f"✅ 训练圆满结束！LoRA 权重已保存在: {output_dir}")

if __name__ == "__main__":
    main()