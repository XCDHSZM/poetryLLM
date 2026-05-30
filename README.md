# 📜 Poetry-LLM-SFT: 基于大语言模型的古典诗词自动生成与格律量化评估系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.37%2B-green)
![Model](https://img.shields.io/badge/Model-Qwen2.5--3B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 项目简介 (Overview)
本项目是一个端到端的大模型指令微调（SFT）工程实践。项目主导了文本生成系统从基础双层 LSTM 架构向基于 Transformer 的前沿大模型（Qwen2.5系列）的演进。
系统专注于中国古典诗词的高质量自动生成，深入剖析了古典诗词的格律特征与语言分布规律，并通过多 GPU 并行训练与 LoRA 参数高效微调技术，实现了从“给定首句”到“完整诗篇”的强逻辑性生成。同时，系统内置了高度专业化的自动化量化评估矩阵。

## ✨ 核心特性 (Key Features)
- **🚀 大模型架构演进**：从传统的 RNN/LSTM 架构升级至主流的 Transformer 大语言模型，引入阿里云开源的 `Qwen2.5-3B-Instruct`。
- **⚡ 高效并行训练**：支持多 GPU 环境（如 Kaggle 双 T4 集群），利用 `Accelerate` 与 `Trainer` 实现半精度（FP16/BF16）分布式混合精度训练。
- **🧠 高质量特征工程**：全链路自动化清洗唐诗 `npz` 语料库，构建严格的训练集与验证集（Train/Val Split），并转化为标准的 Instruction Tuning JSONL 格式。
- **📏 独创格律探针评估**：超越传统的 Loss 评估，创新性地引入古典诗词物理格律探针（字数对齐、严格押韵、平仄交替）与文本重合度指标（BLEU/ROUGE）的交叉评测体系。

## 📂 项目结构 (Project Structure)
```text
Poetry_LLM_Project/
├── data/                    
│   ├── raw/                 # 原始数据集 (tang.npz)
│   └── processed/           # 清洗后的微调数据 (tang_train.jsonl, tang_val.jsonl)
├── models/                  
│   ├── pretrained/          # 预训练大模型权重 (如 Qwen2.5-3B)
│   └── checkpoints/         # LoRA 微调权重与断点
├── src/                     
│   ├── utils/
│   │   └── tools.py         # 动态路径等公共函数
│   ├── data_process/        
│   │   └── make_dataset.py  # 数据清洗与微调格式重构
│   ├── train/               
│   │   └── sft_train.py     # 分布式 LoRA 微调主脚本
│   ├── inference/           
│   │   └── generate.py      # 自动检索最新 Checkpoint 的批量推理脚本
│   └── evaluate/            
│       └── run_eval.py      # 多维度量化评估脚本 (格律 + 统计学指标)
├── requirements.txt         # 核心环境依赖
├── .gitignore               # Git 忽略配置
└── README.md                # 项目文档