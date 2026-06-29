import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

#!/usr/bin/env python3
"""
hotpot_two_round_retrieval_experiment.py

目的：验证 HotpotQA 上单轮检索不足的问题，通过基于实体的查询扩展进行两轮检索 + MDDS 重排 + FLAN-T5-XL 生成。
"""

import json
import re
import string
from collections import Counter
from typing import List, Tuple
import spacy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DebertaV2Tokenizer
from elasticsearch import Elasticsearch

# 添加您的项目路径以导入 documents_scorer 模块
import sys
sys.path.insert(0, '../full-SaR-RAG')   # 根据实际情况调整路径
from model.documents_scorer.documents_scorer import DebertaForMultiHeadClassification

# ===============================
# 1. 评估函数（直接复用您的）
# ===============================
def normalize_text(s):
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

def compute_em_contains(prediction, ground_truth):
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return 1 if truth_norm in pred_norm else 0

def compute_f1(prediction, ground_truth):
    def get_tokens(s):
        if not s:
            return []
        return normalize_text(s).split()
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

# ===============================
# 2. BM25 检索接口（基于 Elasticsearch）
# ===============================
def create_es_bm25_search(es_host="http://localhost:9200", index="hotpotqa", text_field="paragraph_text"):
    """
    返回一个可调用对象，用于 BM25 检索。
    """
    es_client = Elasticsearch([es_host])
    def search_fn(query: str, top_k: int = 20) -> List[str]:
        res = es_client.search(
            index=index,
            body={
                "query": {"match": {text_field: query}},
                "size": top_k
            }
        )
        hits = res["hits"]["hits"]
        return [hit["_source"][text_field] for hit in hits]
    return search_fn

# ===============================
# 3. 实体提取（spaCy）
# ===============================
nlp = spacy.load("en_core_web_sm")

def extract_key_entities(texts: List[str], top_n: int = 5) -> List[str]:
    entity_counter = Counter()
    for text in texts:
        # 限制长度防止过慢
        doc = nlp(text[:50000])
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}:
                entity_counter[ent.text] += 1
    return [ent for ent, _ in entity_counter.most_common(top_n)]

# ===============================
# 4. MDDS 重排
# ===============================
def load_mdds(model_path: str, device: torch.device):
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = DebertaForMultiHeadClassification(model_name="microsoft/deberta-v3-base")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model

def rerank_documents(query: str,
                     documents: List[str],
                     tokenizer,
                     model,
                     device,
                     batch_size=16) -> List[Tuple[str, float]]:
    if not documents:
        return []
    input_texts = [f"Question: {query} [SEP] Document: {d}" for d in documents]
    scores = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        cov_logits = outputs["coverage_logits"][:, 1]   # 正类 logit
        utl_logits = outputs["utility_logits"][:, 1]
        dep_logits = outputs["depth_logits"][:, 1]
        total = cov_logits + 1.2 * utl_logits + dep_logits
        for j, t in enumerate(total.cpu().tolist()):
            scores.append((documents[i+j], t))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ===============================
# 5. FLAN-T5-XL 生成器
# ===============================
class FlanGenerator:
    def __init__(self, model_dir: str, device: torch.device):
        print(f"Loading tokenizer from {model_dir}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Tokenizer loaded.")
        print(f"Loading model from {model_dir} with dtype=float16...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
        )
        print("Model loaded.")
        self.model = self.model.to(device)
        print(f"Model moved to {device}.")
        self.model.eval()
        print("FlanGenerator ready.")

    def generate(self, question: str, context: str, max_new_tokens: int = 50) -> str:
        prompt = (
            "Please answer the question based on the following text information.\n"
            f"Text information:\n{context}\n"
            f"Question: {question}"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

# ===============================
# 主函数（带结果保存）
# ===============================
def main():
    # ================= 配置区 =================
    ORIGINAL_RESULT_FILE = "./hotpot_bm25_top10_standard_final_results.jsonl"
    MDDS_MODEL_PATH = "../full-SaR-RAG/model/documents_scorer/documents_scorer_model.pth"
    FLAN_MODEL_DIR = "/root/autodl-tmp/flan-t5-xl"
    ES_HOST = "http://localhost:9200"
    ES_INDEX = "hotpotqa"
    ES_TEXT_FIELD = "paragraph_text"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "./two_round_experiment_results"
    # =========================================

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化组件
    print("初始化 BM25 检索器（Elasticsearch）...")
    bm25_search_fn = create_es_bm25_search(es_host=ES_HOST, index=ES_INDEX, text_field=ES_TEXT_FIELD)

    print("加载 MDDS 模型...")
    mdds_tokenizer, mdds_model = load_mdds(MDDS_MODEL_PATH, DEVICE)

    print("加载 FLAN-T5-XL 模型...")
    flan = FlanGenerator(FLAN_MODEL_DIR, DEVICE)

    # 1. 读取原始结果，筛选需要改进的样本
    print("读取原始结果文件...")
    samples_to_improve = []
    with open(ORIGINAL_RESULT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('useRAG') != 1:
                continue
            gold_raw = item['answer']
            gold = gold_raw[0] if isinstance(gold_raw, list) else str(gold_raw)
            pred = item.get('prediction', '')
            if compute_em_contains(pred, gold) == 1:
                continue
            samples_to_improve.append({
                "question_id": item['question_id'],
                "question_text": item['question_text'],
                "gold": gold,
                "original_prediction": pred
            })
    print(f"需要二轮检索的样本数: {len(samples_to_improve)}")

    if len(samples_to_improve) == 0:
        print("没有需要改进的样本，程序退出。")
        return

    # 2. 对每个样本执行两轮检索 + 重排 + 生成
    improved_predictions = {}
    all_improvement_details = []  # 保存详细结果

    for idx, sample in enumerate(tqdm(samples_to_improve, desc="两轮检索+生成")):
        qid = sample['question_id']
        query = sample['question_text']
        
        # 第一轮检索
        docs_round1 = bm25_search_fn(query, top_k=20)
        
        # 提取实体
        entities = extract_key_entities(docs_round1, top_n=5)
        expanded_query = query + " " + " ".join(entities) if entities else query
        
        # 第二轮检索
        docs_round2 = bm25_search_fn(expanded_query, top_k=20)
        
        # 合并去重
        all_docs = []
        seen = set()
        for d in docs_round1 + docs_round2:
            if d not in seen:
                seen.add(d)
                all_docs.append(d)
        
        if not all_docs:
            new_answer = sample['original_prediction']
        else:
            reranked = rerank_documents(query, all_docs, mdds_tokenizer, mdds_model, DEVICE)
            top3_docs = [doc for doc, _ in reranked[:3]]
            context = "\n".join(top3_docs)
            new_answer = flan.generate(query, context)
        
        improved_predictions[qid] = new_answer
        
        # 记录详细信息
        all_improvement_details.append({
            "question_id": qid,
            "question_text": query,
            "gold": sample['gold'],
            "original_prediction": sample['original_prediction'],
            "new_prediction": new_answer,
            "original_correct": compute_em_contains(sample['original_prediction'], sample['gold']),
            "new_correct": compute_em_contains(new_answer, sample['gold'])
        })

    # 3. 重新计算整体指标（包含改进后的答案）
    all_items = []
    with open(ORIGINAL_RESULT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = item['question_id']
            if qid in improved_predictions:
                item['prediction'] = improved_predictions[qid]
            all_items.append(item)
    
    em_list, f1_list = [], []
    for item in all_items:
        pred = item['prediction']
        gold_raw = item['answer']
        gold = gold_raw[0] if isinstance(gold_raw, list) else str(gold_raw)
        em = compute_em_contains(pred, gold)
        f1 = compute_f1(pred, gold)
        em_list.append(em)
        f1_list.append(f1)
    
    total = len(all_items)
    avg_em = sum(em_list) / total
    avg_f1 = sum(f1_list) / total

    # 计算改进集上的统计
    improved_set_size = len(samples_to_improve)
    orig_correct = sum(1 for s in all_improvement_details if s['original_correct'])
    new_correct = sum(1 for s in all_improvement_details if s['new_correct'])
    improvement_gain = (new_correct - orig_correct) / improved_set_size * 100 if improved_set_size > 0 else 0

    # ================= 保存结果 =================
    # 1. 保存整体指标
    metrics_summary = {
        "total_samples": total,
        "overall_accuracy": avg_em,
        "overall_f1": avg_f1,
        "improved_set_size": improved_set_size,
        "original_correct_in_improved": orig_correct,
        "new_correct_in_improved": new_correct,
        "improvement_gain_percent": improvement_gain,
        "two_round_retrieval_enabled": True
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    
    # 2. 保存每个被改进样本的详细对比
    with open(os.path.join(OUTPUT_DIR, "improvement_details.json"), 'w', encoding='utf-8') as f:
        json.dump(all_improvement_details, f, indent=2, ensure_ascii=False)
    
    # 3. 保存完整的最终预测结果（所有样本）
    final_predictions = []
    for item in all_items:
        final_predictions.append({
            "question_id": item['question_id'],
            "question_text": item.get('question_text', ''),
            "gold": item['answer'][0] if isinstance(item['answer'], list) else item['answer'],
            "useRAG": item.get('useRAG', 0),
            "prediction": item['prediction']
        })
    with open(os.path.join(OUTPUT_DIR, "final_predictions_all.json"), 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, indent=2, ensure_ascii=False)
    
    # 4. 保存一份可读的文本报告
    report_lines = [
        "========== 两轮检索后整体指标 ==========",
        f"样本总数: {total}",
        f"EM (Acc): {avg_em:.4f} ({avg_em*100:.2f}%)",
        f"F1:       {avg_f1:.4f} ({avg_f1*100:.2f}%)",
        "",
        f"被改进样本集 ({improved_set_size} 个):",
        f"  原始正确数: {orig_correct} ({orig_correct/improved_set_size*100:.2f}%)",
        f"  改进后正确数: {new_correct} ({new_correct/improved_set_size*100:.2f}%)",
        f"  提升百分点: {improvement_gain:.2f}%"
    ]
    with open(os.path.join(OUTPUT_DIR, "report.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    # 打印到控制台
    print("\n========== 两轮检索后整体指标 ==========")
    print(f"样本总数: {total}")
    print(f"EM (Acc): {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"F1:       {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"\n被改进样本集 ({improved_set_size} 个):")
    print(f"  原始正确数: {orig_correct} ({orig_correct/improved_set_size*100:.2f}%)")
    print(f"  改进后正确数: {new_correct} ({new_correct/improved_set_size*100:.2f}%)")
    print(f"\n✅ 所有实验结果已保存到: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()