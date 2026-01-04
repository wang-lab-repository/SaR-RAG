import json
import re
import string

def normalize_text(text):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(s):
        return re.sub(r'\b(a|an|the)\b', ' ', s)

    def white_space_fix(s):
        return ' '.join(s.split())

    def remove_punc(s):
        return s.translate(str.maketrans('', '', string.punctuation))

    if text is None:
        return ""
    text = text.lower()
    text = remove_articles(text)
    text = remove_punc(text)
    text = white_space_fix(text)
    return text

# Step 1: Load TriviaQA standard answers
standard_answers = {}
with open("../trivia_qa_dev_standard.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        q_id = item["question_id"]
        ans_obj = item["answer"]

        # Collect all answer variants: value + aliases
        candidates = set()
        candidates.add(ans_obj["value"])
        if "aliases" in ans_obj and isinstance(ans_obj["aliases"], list):
            candidates.update(ans_obj["aliases"])

        # Normalize each candidate and filter empty
        normalized_answers = []
        for cand in candidates:
            norm = normalize_text(cand)
            if norm != "":
                normalized_answers.append(norm)

        # Remove duplicates after normalization (e.g., "Sunset Blvd." vs "Sunset Blvd")
        normalized_answers = list(set(normalized_answers))
        standard_answers[q_id] = normalized_answers

# Step 2: Evaluate Hit@k with substring containment
hit_counts = [0] * 20  # Hit@1 to Hit@20
total_samples = 0

with open("sampled_rerank_results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        q_id = item["id"]  # sampled_results.jsonl still uses "id"
        predictions = item["prediction"]  # list of 20 strings

        if q_id not in standard_answers:
            continue  # skip if not in dev set

        gold_list = standard_answers[q_id]  # list of normalized gold strings
        total_samples += 1

        # Preprocess predictions
        normalized_preds = []
        for pred in predictions:
            # Mark as invalid if starts with "NOT GIVEN" (case-insensitive)
            if isinstance(pred, str) and pred.strip().lower().startswith("not given"):
                normalized_preds.append(None)
            else:
                norm_pred = normalize_text(pred)
                normalized_preds.append(norm_pred if norm_pred != "" else None)

        # Check Hit@1 to Hit@20
        for k in range(1, 21):
            hit = False
            for i in range(k):
                np = normalized_preds[i]
                if np is None:
                    continue
                # Substring containment: does np contain any gold answer?
                for gold in gold_list:
                    if gold == "":
                        continue
                    if gold in np:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                hit_counts[k - 1] += 1

# Step 3: Output results
print(f"Total evaluated samples: {total_samples}")
for k in range(1, 21):
    hit_at_k = hit_counts[k - 1] / total_samples if total_samples > 0 else 0
    print(f"Hit@{k}: {hit_at_k:.4f} ({hit_counts[k - 1]}/{total_samples})")

'''
Total evaluated samples: 500
Hit@1: 0.7060 (353/500)
Hit@2: 0.7800 (390/500)
Hit@3: 0.8240 (412/500)
Hit@4: 0.8520 (426/500)
Hit@5: 0.8600 (430/500)
Hit@6: 0.8800 (440/500)
Hit@7: 0.8880 (444/500)
Hit@8: 0.8940 (447/500)
Hit@9: 0.8980 (449/500)
Hit@10: 0.9040 (452/500)
Hit@11: 0.9040 (452/500)
Hit@12: 0.9100 (455/500)
Hit@13: 0.9100 (455/500)
Hit@14: 0.9100 (455/500)
Hit@15: 0.9140 (457/500)
Hit@16: 0.9140 (457/500)
Hit@17: 0.9140 (457/500)
Hit@18: 0.9180 (459/500)
Hit@19: 0.9240 (462/500)
Hit@20: 0.9240 (462/500)
'''