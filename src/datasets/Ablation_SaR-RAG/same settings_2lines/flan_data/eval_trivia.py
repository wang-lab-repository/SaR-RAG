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


def load_trivia_answers(answer_file):
    """
    加载 TriviaQA 答案文件，返回 {question_id: [all_gold_answers]}
    包含 value + aliases（去重、非空、字符串化）
    """
    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["question_id"]
            ans_obj = item["answer"]

            gold_set = set()

            # 添加主 value
            if "value" in ans_obj and ans_obj["value"] is not None:
                gold_set.add(str(ans_obj["value"]).strip())

            # 添加 aliases（如果存在）
            if "aliases" in ans_obj and isinstance(ans_obj["aliases"], list):
                for alias in ans_obj["aliases"]:
                    if alias is not None:
                        gold_set.add(str(alias).strip())

            # 转为列表（顺序无关，后续取 max）
            answers[qid] = list(gold_set)
    return answers


def main():
    pred_file = "./trivia_qa_dev_results.jsonl"
    answer_file = "./trivia_qa_dev_standard.jsonl"

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测文件 {pred_file} 不存在")
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"答案文件 {answer_file} 不存在")

    gold_answers = load_trivia_answers(answer_file)

    em_list, f1_list = [], []
    total = 0
    missing_ids = []
    duplicate_ids = set()
    seen_ids = set()          # 用于去重：记录已经处理过的 id

    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["id"]

            # 去重：如果这个 id 已经处理过，跳过并记录
            if qid in seen_ids:
                duplicate_ids.add(qid)
                continue
            seen_ids.add(qid)

            use_rag = item.get("useRAG", 0)
            pred_raw = item.get("answer", "")

            # ===== 根据 useRAG 正确处理预测字段 =====
            if use_rag == 1 and isinstance(pred_raw, list):
                # 多候选情况：pred_raw 是一个列表，例如 ["ans1", "ans2", "ans3"]
                # 计算 EM：将所有候选答案标准化后拼接，检查任一 gold 是否在其中
                normalized_preds = [normalize_text(str(p) if p is not None else "") for p in pred_raw]
                combined_pred = " ".join(normalized_preds)
                combined_norm = normalize_text(combined_pred)
                best_em = 0
                for gold in gold_answers.get(qid, []):
                    if normalize_text(gold) in combined_norm:
                        best_em = 1
                        break

                # 计算 F1：每个候选与所有 gold 的最佳 F1，再取所有候选的最大值
                best_f1 = 0.0
                for single_pred in pred_raw:
                    single_pred_str = str(single_pred) if single_pred is not None else ""
                    for gold in gold_answers.get(qid, []):
                        f1_candidate = compute_f1(single_pred_str, gold)
                        if f1_candidate > best_f1:
                            best_f1 = f1_candidate
            else:
                # useRAG == 0 或 prediction 不是列表：按单字符串处理
                pred_str = str(pred_raw).strip()
                gold_list = gold_answers.get(qid, [])
                if not gold_list:
                    # 没有 gold 答案，跳过该样本（不计入总数）
                    continue
                # 计算 EM 和 F1（取所有 gold 中的最大值）
                em_scores = [compute_em_contains(pred_str, gold) for gold in gold_list]
                f1_scores = [compute_f1(pred_str, gold) for gold in gold_list]
                best_em = max(em_scores)
                best_f1 = max(f1_scores)

            # 跳过没有 gold 的样本（对于多候选情况也需要检查）
            if qid not in gold_answers:
                missing_ids.append(qid)
                continue

            em_list.append(best_em)
            f1_list.append(best_f1)
            total += 1

    # 输出重复警告
    if duplicate_ids:
        print(f"⚠️ 警告：预测文件中存在 {len(duplicate_ids)} 个重复 ID，已保留首次出现，跳过后续重复行。")
        print(f"   重复 ID 示例（最多5个）：{list(duplicate_ids)[:5]}")
    if missing_ids:
        print(f"⚠️ 警告：在答案文件中未找到 {len(missing_ids)} 个预测 ID，已跳过。示例：{missing_ids[:3]}")

    avg_em = sum(em_list) / total if total else 0
    avg_f1 = sum(f1_list) / total if total else 0

    print(f"\n📊 TriviaQA 评估完成！共 {total} 个有效样本（原始预测行数 {len(seen_ids)+len(duplicate_ids)}，去重后 {len(seen_ids)}，有效评估 {total}）\n")
    print(f"EM (子串匹配 Acc): {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print(f"F1 (词级别 F1):   {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")


if __name__ == "__main__":
    main()