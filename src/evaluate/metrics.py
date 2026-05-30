import pypinyin
import re
def evaluate_poetry_format(generated_poem):
    """
    评估生成的诗词是否符合基础格律
    返回一个字典，包含各项得分 (0-100分)
    """
    scores = {}
    # 1. 清洗并按句号/逗号分句
    sentences = re.split(r'[，。！？]', generated_poem.strip())
    sentences = [s for s in sentences if len(s) > 0]
    if len(sentences) < 2:
        return {"格式分": 0, "字数对齐分": 0, "押韵分": 0}
    # 2. 字数对齐检测 (五言或七言)
    # 检查所有句子的长度是否一致，且是否为 5 或 7
    lengths = [len(s) for s in sentences]
    is_aligned = all(l == lengths[0] for l in lengths)
    is_standard_length = lengths[0] in [5, 7]
    
    if is_aligned and is_standard_length:
        scores["字数对齐分"] = 100
    else:
        scores["字数对齐分"] = 0

    # 3. 押韵检测 (检查偶数句的韵脚)
    # 取偶数句的最后一个字
    rhyme_chars = [sentences[i][-1] for i in range(1, len(sentences), 2)]
    
    if len(rhyme_chars) >= 2:
        # 获取拼音的韵母 (finals)
        pinyins = pypinyin.pinyin(rhyme_chars, style=pypinyin.FINALS)
        # 简单校验：提取的核心韵母是否一致
        rhymes = [p[0] for p in pinyins]
        if len(set(rhymes)) == 1: # 所有偶数句韵母完全一致
            scores["押韵分"] = 100
        else:
            scores["押韵分"] = int((1 - (len(set(rhymes)) - 1) / len(rhymes)) * 100)
    else:
         scores["押韵分"] = 100 # 只有两句时，首句和次句不一定严格要求同韵

    return scores