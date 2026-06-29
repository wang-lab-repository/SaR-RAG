import json
import re
import string
import random
from pathlib import Path

# ========== 相对路径配置（基于当前工作目录） ==========
# 当前目录假定为：datasets/gpt_claude_sampled200/llm_self_scores/
RESULT_FILE = Path("../../datasets/trivia_qa/FLAN/without Retrieval/trivia_qa_dev_noRag_results.jsonl")
GOLD_FILE = Path("../../trivia_qa/trivia_qa_dev_standard.jsonl")

OUTPUT_FILE = Path("trivia_qa_dev_noRag_results_with_canAnswer.jsonl")
SAMPLE_FILE = Path("sampled_balanced_100_each.jsonl")

random.seed(42)

# ========== 标准化函数 ==========
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)

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

# ========== 读取标准答案，合并重复 question_id 的黄金答案 ==========
print("正在加载标准答案（合并重复 id 的答案）...")
gold_map = {}  # {qid: set of normalized gold strings}
with open(GOLD_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qid = data["question_id"]
        ans_obj = data["answer"]

        raw_candidates = []
        if "value" in ans_obj and ans_obj["value"]:
            raw_candidates.append(ans_obj["value"])
        if "aliases" in ans_obj and ans_obj["aliases"]:
            raw_candidates.extend(ans_obj["aliases"])

        # 临时标准化这批候选答案
        norm_candidates = [normalize_text(c) for c in raw_candidates if c]
        norm_candidates = [c for c in norm_candidates if c]  # 去空

        if qid not in gold_map:
            gold_map[qid] = set()
        gold_map[qid].update(norm_candidates)

print(f"加载完成，共 {len(gold_map)} 个唯一问题 ID")

# ========== 处理结果文件，按 id 去重（保留首次出现） ==========
print("正在处理结果文件（按 id 去重，保留首次出现）...")
records_dict = {}  # {id: record}
with open(RESULT_FILE, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        record = json.loads(line)
        qid = record.get("id")
        if qid is None:
            continue
        if qid not in records_dict:   # 保留首次出现
            records_dict[qid] = record

print(f"结果文件原始行数（可能有重复 id）: 未知，去重后唯一 id 数: {len(records_dict)}")

# ========== 为每条记录添加 canAnswer 字段 ==========
all_records = []
for qid, record in records_dict.items():
    raw_answer = record.get("answer", "")
    norm_answer = normalize_text(raw_answer)

    can_answer = 0
    if qid in gold_map:
        for gold in gold_map[qid]:
            if gold and gold in norm_answer:
                can_answer = 1
                break

    record["canAnswer"] = can_answer
    all_records.append(record)

# 输出完整去重后的结果文件（按 id 顺序，可选）
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    for rec in all_records:
        f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"处理完成，结果已保存至 {OUTPUT_FILE.resolve()}")
total = len(all_records)
cnt_1 = sum(r['canAnswer'] for r in all_records)
cnt_0 = total - cnt_1
print(f"总共 {total} 条唯一记录，canAnswer=1: {cnt_1} 条，canAnswer=0: {cnt_0} 条")

# ========== 随机采样 canAnswer=0 和 canAnswer=1 各 100 条 ==========
records_1 = [r for r in all_records if r['canAnswer'] == 1]
records_0 = [r for r in all_records if r['canAnswer'] == 0]

sample_size = 100
sampled_1 = random.sample(records_1, min(sample_size, len(records_1)))
sampled_0 = random.sample(records_0, min(sample_size, len(records_0)))
sampled = sampled_1 + sampled_0
random.shuffle(sampled)

print(f"采样结果：canAnswer=1 采了 {len(sampled_1)} 条，canAnswer=0 采了 {len(sampled_0)} 条")

with open(SAMPLE_FILE, 'w', encoding='utf-8') as f:
    for rec in sampled:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"采样结果已保存至 {SAMPLE_FILE.resolve()}")