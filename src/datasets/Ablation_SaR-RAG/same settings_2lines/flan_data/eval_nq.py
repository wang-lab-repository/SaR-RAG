import json
import re
import string
from collections import Counter
import os


def normalize_text(text):
    """标准化文本：小写、去标点、去冠词(a/an/the)、去多余空格"""
    if text is None:
        return ""
    # 转为小写
    text = text.lower()
    # 移除冠词
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # 移除标点
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 合并多余空格
    text = ' '.join(text.split())
    return text


def get_tokens(s):
    return normalize_text(s).split() if s else []


def compute_f1(prediction, ground_truth):
    """计算单个预测与单个标准答案的词级别 F1"""
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    intersection = pred_counter & truth_counter
    overlap = sum(intersection.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_single_sample(prediction, gold_answers):
    """
    评估单个样本：返回 (em_score, f1_score)
    - prediction: 可能是字符串或字符串列表
    - gold_answers: 字符串列表（标准答案的多个变体）
    """
    # ---------- 处理预测（可能为列表）----------
    if isinstance(prediction, list):
        # 1. 将列表拼接为一个大字符串（用于 EM 子串匹配）
        pred_merged = ' '.join(str(p) if p is not None else "" for p in prediction)
        # 2. 保留原始列表用于 F1 候选取优
        pred_list = [str(p) if p is not None else "" for p in prediction]
    else:
        pred_merged = str(prediction) if prediction is not None else ""
        pred_list = [pred_merged]

    # ---------- 计算 EM（子串包含）----------
    norm_pred_merged = normalize_text(pred_merged)
    em = 0
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if norm_gold in norm_pred_merged:
            em = 1
            break

    # ---------- 计算 F1（对多个候选预测取最大，对多个标准答案取最大）----------
    best_overall_f1 = 0.0
    for pred_str in pred_list:
        # 当前预测与所有标准答案的最大 F1
        best_for_this_pred = max((compute_f1(pred_str, gold) for gold in gold_answers), default=0.0)
        if best_for_this_pred > best_overall_f1:
            best_overall_f1 = best_for_this_pred

    return em, best_overall_f1


def main():
    # 文件路径（请根据实际情况修改）
    pred_file = "NQ-open_dev_results.jsonl"      # 模型预测文件
    gold_file = "NQ-open_dev_standard.jsonl"     # 标准答案文件

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"标准答案文件不存在: {gold_file}")

    # 读取预测文件，按 id 索引
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item['id']
            pred = item.get('answer', "")      # 预测内容（可能是 str 或 list）
            predictions[qid] = pred

    # 读取标准答案文件，按 id 索引
    gold_answers = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item['id']
            ans = item.get('answer', [])
            # 确保是字符串列表
            if isinstance(ans, str):
                ans = [ans]
            elif not isinstance(ans, list):
                ans = [str(ans)]
            gold_answers[qid] = ans

    # 只评估两个文件中都存在的 id
    common_ids = set(predictions.keys()) & set(gold_answers.keys())
    print(f"预测文件包含 {len(predictions)} 条，标准文件包含 {len(gold_answers)} 条")
    print(f"共同 ID 数量: {len(common_ids)}")

    if not common_ids:
        print("错误：没有匹配的 ID，请检查文件格式。")
        return

    em_scores = []
    f1_scores = []

    for qid in common_ids:
        pred = predictions[qid]
        golds = gold_answers[qid]
        em, f1 = evaluate_single_sample(pred, golds)
        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print("\n========== 评估结果 ==========")
    print(f"样本总数: {len(em_scores)}")
    print(f"EM (子串匹配准确率): {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"F1 (最佳预测‑最佳答案): {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print("==============================\n")


if __name__ == "__main__":
    main()