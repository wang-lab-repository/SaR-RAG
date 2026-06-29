import json
import re
import string
from collections import Counter
import os


def normalize_text(s):
    """标准化文本：转小写、去标点、去多余空格"""
    if s is None:
        return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def get_tokens(s):
    return normalize_text(s).split() if s else []


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
    return 2 * (precision * recall) / (precision + recall)


def compute_em_contains(prediction, ground_truth):
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return 1 if truth_norm in pred_norm else 0


def load_hotpot_answers(answer_file):
    """HotpotQA 答案文件：每行 {"_id": "...", "answer": "..."}"""
    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["_id"]
            ans = item.get("answer", "")
            answers[qid] = str(ans).strip()
    return answers


def main():
    pred_file = "./hotpot_dev_results.jsonl"
    answer_file = "./hotpot_dev_fullwiki_v1.jsonl"

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测文件 {pred_file} 不存在")
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"答案文件 {answer_file} 不存在")

    gold_answers = load_hotpot_answers(answer_file)

    em_list, f1_list = [], []
    total = 0
    missing_ids = []

    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["id"]
            pred_raw = item.get("prediction", "")

            if qid not in gold_answers:
                missing_ids.append(qid)
                continue

            gold = gold_answers[qid]

            # 核心修改：根据 pred_raw 的类型决定处理方式
            if isinstance(pred_raw, list):
                # 多候选情况：pred_raw 是一个列表，例如 ["answer1", "answer2", "answer3"]
                # 1. EM：合并所有候选并检查子串
                merged_pred = ' '.join(str(p) if p is not None else "" for p in pred_raw)
                best_em = compute_em_contains(merged_pred, gold)
                # 2. F1：取所有候选与 gold 的最大 F1
                best_f1 = 0.0
                for p in pred_raw:
                    p_str = str(p) if p is not None else ""
                    f1 = compute_f1(p_str, gold)
                    if f1 > best_f1:
                        best_f1 = f1
            else:
                # 单字符串情况
                pred_str = str(pred_raw).strip()
                best_em = compute_em_contains(pred_str, gold)
                best_f1 = compute_f1(pred_str, gold)

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