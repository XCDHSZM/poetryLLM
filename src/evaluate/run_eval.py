import os
import json
import re
import numpy as np
import pypinyin
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 为了能顺利导入动态路径，临时将项目根目录加入系统路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.tools import get_project_root

# ==========================================
# 1. 物理格律探针 (字数、押韵、平仄)
# ==========================================
def evaluate_poetry_format(generated_poem):
    scores = {"字数对齐分": 0, "押韵分": 0, "平仄交替分": 0}
    
    # 清洗并分句 (去除空格等多余字符)
    generated_poem = generated_poem.replace(" ", "").strip()
    sentences = re.split(r'[，。！？]', generated_poem)
    sentences = [s for s in sentences if len(s) > 0]
    
    if len(sentences) < 2:
        return scores

    # 1. 字数对齐检测
    lengths = [len(s) for s in sentences]
    is_aligned = all(l == lengths[0] for l in lengths)
    is_standard_length = lengths[0] in [5, 7]
    if is_aligned and is_standard_length:
        scores["字数对齐分"] = 100

    # 2. 押韵检测 (忽略声调)
    rhyme_chars = [sentences[i][-1] for i in range(1, len(sentences), 2)]
    if len(rhyme_chars) >= 2:
        pinyins = pypinyin.pinyin(rhyme_chars, style=pypinyin.Style.FINALS, errors='ignore')
        rhymes = [p[0] for p in pinyins if p]
        if len(rhymes) > 0:
            most_common_rhyme = max(set(rhymes), key=rhymes.count)
            match_ratio = rhymes.count(most_common_rhyme) / len(rhymes)
            scores["押韵分"] = int(match_ratio * 100)
    else:
         scores["押韵分"] = 100 

    # 3. 平仄交替检测
    pingze_score_list = []
    for sentence in sentences:
        tones = pypinyin.pinyin(sentence, style=pypinyin.Style.TONE3, errors='ignore')
        pz_pattern = []
        for t in tones:
            if not t: continue
            tone_num = re.search(r'\d', t[0])
            if tone_num:
                tone_val = int(tone_num.group())
                pz_pattern.append('平' if tone_val in [1, 2] else '仄')
            else:
                pz_pattern.append('平')
                
        if len(pz_pattern) > 1:
            alternation_count = sum(1 for i in range(len(pz_pattern)-1) if pz_pattern[i] != pz_pattern[i+1])
            pingze_score_list.append(alternation_count / (len(pz_pattern) - 1))
            
    if pingze_score_list:
        avg_alternation = sum(pingze_score_list) / len(pingze_score_list)
        scores["平仄交替分"] = int(avg_alternation * 100)

    return scores

# ==========================================
# 2. 文本重合度指标 (BLEU & ROUGE)
# ==========================================
def calculate_overlap_metrics(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    smooth = SmoothingFunction().method1
    
    # 中文按字切分，过滤掉可能存在的空格
    ref_chars = list(reference.replace(" ", "").strip())
    hyp_chars = list(hypothesis.replace(" ", "").strip())
    
    # 防止空字符串报错
    if not ref_chars or not hyp_chars:
         return {"BLEU": 0.0, "ROUGE-L": 0.0}

    bleu = sentence_bleu([ref_chars], hyp_chars, smoothing_function=smooth)
    
    ref_str = " ".join(ref_chars)
    hyp_str = " ".join(hyp_chars)
    rouge_result = scorer.score(ref_str, hyp_str)
    
    return {
        "BLEU": bleu * 100,
        "ROUGE-L": rouge_result['rougeL'].fmeasure * 100
    }

# ==========================================
# 3. 主程序：加载真实推理结果并生成报告
# ==========================================
def main():
    root_dir = get_project_root()
    # 假设你的推理脚本跑完后，将结果保存在了这个文件里
    # 文件格式需满足每一行是一个 JSON: {"instruction": "...", "target": "原诗后续", "generated": "AI生成后续"}
    results_path = os.path.join(root_dir, "data", "processed", "eval_results.jsonl")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"❌ 找不到推理结果文件：{results_path}\n请先运行批量推理脚本生成此文件！")

    print("="*50)
    print("🚀 启动唐诗大模型自动量化评估 (读取真实推理数据)")
    print("="*50)

    all_format_scores = {"字数对齐分": [], "押韵分": [], "平仄交替分": []}
    all_overlap_scores = {"BLEU": [], "ROUGE-L": []}
    
    valid_count = 0
    error_count = 0

    with open(results_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                target = item.get("target", "")
                generated = item.get("generated", "")
                
                if not target or not generated:
                    continue # 跳过空数据
                
                # 计算格律
                format_scores = evaluate_poetry_format(generated)
                for k, v in format_scores.items():
                    all_format_scores[k].append(v)
                    
                # 计算重合度
                overlap_scores = calculate_overlap_metrics(target, generated)
                for k, v in overlap_scores.items():
                    all_overlap_scores[k].append(v)
                    
                valid_count += 1
                
                # 打印一点进度，防止感觉程序卡死
                if valid_count % 100 == 0:
                     print(f"已评估 {valid_count} 条诗词数据...")

            except json.JSONDecodeError:
                error_count += 1
                print(f"⚠️ 警告：第 {line_idx + 1} 行 JSON 解析失败，已跳过。")
            except Exception as e:
                error_count += 1
                print(f"⚠️ 警告：评估第 {line_idx + 1} 行时出错 ({e})，已跳过。")

    if valid_count == 0:
        print("❌ 没有成功评估任何有效数据，请检查 eval_results.jsonl 文件格式。")
        return

    # 打印最终实验报告
    print("\n" + "="*50)
    print(f"📊 【最终实验评估结果报告】 (共处理有效样本: {valid_count} 条)")
    if error_count > 0:
         print(f"⚠️ 注意：跳过了 {error_count} 条异常数据。")
    print("-" * 30)
    
    print("一、 文本重合度 (模型还原记忆能力)")
    for metric, scores in all_overlap_scores.items():
        avg = np.mean(scores)
        print(f"  ▶ {metric:8} : {avg:.2f}")
        
    print("\n二、 古典格律指标 (模型生成规则遵循能力)")
    for metric, scores in all_format_scores.items():
        avg = np.mean(scores)
        print(f"  ▶ {metric:8} : {avg:.2f}")
        
    print("="*50)
    print("💡 提示：将此数据填入实验报告的表格中，并可生成雷达图增加专业度。")

if __name__ == "__main__":
    main()