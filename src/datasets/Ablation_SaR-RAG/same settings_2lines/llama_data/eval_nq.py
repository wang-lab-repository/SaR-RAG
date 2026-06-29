import json
import re
import string
from collections import Counter
import os


def normalize_text(s):
    """标准化文本：转小写、去标点、去多余空格"""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_text(s).split()


def compute_f1(prediction, ground_truth):
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
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_em_contains(prediction, ground_truth):
    """子串匹配：标准化后，gold 是否出现在 prediction 中"""
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return 1 if truth_norm in pred_norm else 0


def load_nq_answers(answer_file):
    """加载 NQ-open 格式的答案文件，返回 {id: [answer1, answer2, ...]}"""
    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["id"]
            ans_list = item["answer"]  # list of strings
            answers[qid] = [str(a).strip() for a in ans_list if a is not None]
    return answers


def main():
    pred_file = "./NQ-open_dev_results_final.jsonl"
    answer_file = "./NQ-open_dev_with_id.jsonl"

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测文件 {pred_file} 不存在")
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"答案文件 {answer_file} 不存在")

    gold_answers = load_nq_answers(answer_file)

    em_list, f1_list = [], []
    total = 0
    missing_ids = []

    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["id"]
            use_rag = item.get("useRAG", 0)
            pred_raw = item.get("prediction", "")

            # 根据 useRAG 处理预测
            if use_rag == 1 and isinstance(pred_raw, list):
                # 多候选情况：pred_raw 是一个列表，例如 ["ans1", "ans2", "ans3"]
                # 1. EM：将所有候选答案标准化后拼接，检查任一 gold 是否在其中
                merged_pred = ' '.join(str(p) if p is not None else "" for p in pred_raw)
                norm_merged = normalize_text(merged_pred)
                best_em = 0
                gold_list = gold_answers.get(qid, [])
                for gold in gold_list:
                    if normalize_text(gold) in norm_merged:
                        best_em = 1
                        break

                # 2. F1：分别计算每个候选与所有 gold 的最佳 F1，取最大值
                best_f1 = 0.0
                for single_pred in pred_raw:
                    single_pred_str = str(single_pred) if single_pred is not None else ""
                    for gold in gold_list:
                        f1_candidate = compute_f1(single_pred_str, gold)
                        if f1_candidate > best_f1:
                            best_f1 = f1_candidate
            else:
                # useRAG == 0 或预测不是列表：按单字符串处理
                pred_str = str(pred_raw).strip()
                gold_list = gold_answers.get(qid, [])
                if not gold_list:
                    continue
                # 计算 EM 和 F1（取所有 gold 中的最大值）
                em_scores = [compute_em_contains(pred_str, gold) for gold in gold_list]
                f1_scores = [compute_f1(pred_str, gold) for gold in gold_list]
                best_em = max(em_scores)
                best_f1 = max(f1_scores)

            if qid not in gold_answers:
                missing_ids.append(qid)
                continue

            em_list.append(best_em)
            f1_list.append(best_f1)
            total += 1

    if missing_ids:
        print(f"⚠️ 警告：在答案文件中未找到 {len(missing_ids)} 个预测 ID，已跳过。示例：{missing_ids[:3]}")

    avg_em = sum(em_list) / total if total else 0
    avg_f1 = sum(f1_list) / total if total else 0

    print(f"\n📊 评估完成！共 {total} 个有效样本\n")
    print(f"EM (子串匹配 Acc): {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print(f"F1 (词级别 F1):   {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")


if __name__ == "__main__":
    main()