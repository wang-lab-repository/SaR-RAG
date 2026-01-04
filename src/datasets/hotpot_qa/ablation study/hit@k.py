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

# Step 1: Load standard answers and normalize them
standard_answers = {}
with open("../hotpot_dev_fullwiki_v1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        # Filter out empty answers (just in case)
        normalized_answers = [normalize_text(item["answer"])]
        standard_answers[item["_id"]] = normalized_answers

# Step 2: Evaluate Hit@k with SUBSTRING CONTAINMENT
hit_counts = [0] * 20  # Hit@1 to Hit@20
total_samples = 0

with open("sampled_results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        q_id = item["id"]
        predictions = item["prediction"]  # list of 20 strings

        if q_id not in standard_answers:
            continue  # skip if ID not found

        gold_list = standard_answers[q_id]  # list of normalized gold answers (non-empty)
        total_samples += 1

        # Preprocess predictions: normalize valid ones
        normalized_preds = []
        for pred in predictions:
            # Treat as invalid if starts with "NOT GIVEN" (case-insensitive)
            if isinstance(pred, str) and pred.strip().lower().startswith("not given"):
                normalized_preds.append(None)
            else:
                norm_pred = normalize_text(pred)
                normalized_preds.append(norm_pred if norm_pred != "" else None)

        # For each k from 1 to 20
        for k in range(1, 21):
            hit = False
            # Check first k predictions
            for i in range(k):
                np = normalized_preds[i]
                if np is None:
                    continue
                # Check if np CONTAINS any gold answer as substring
                for gold in gold_list:
                    if gold == "":  # skip empty
                        continue
                    if gold in np:  # ← substring containment
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
Hit@1: 0.2460 (123/500)
Hit@2: 0.2960 (148/500)
Hit@3: 0.3340 (167/500)
Hit@4: 0.3520 (176/500)
Hit@5: 0.3940 (197/500)
Hit@6: 0.4120 (206/500)
Hit@7: 0.4200 (210/500)
Hit@8: 0.4240 (212/500)
Hit@9: 0.4340 (217/500)
Hit@10: 0.4500 (225/500)
Hit@11: 0.4520 (226/500)
Hit@12: 0.4540 (227/500)
Hit@13: 0.4560 (228/500)
Hit@14: 0.4580 (229/500)
Hit@15: 0.4620 (231/500)
Hit@16: 0.4680 (234/500)
Hit@17: 0.4740 (237/500)
Hit@18: 0.4820 (241/500)
Hit@19: 0.4880 (244/500)
Hit@20: 0.4960 (248/500)
'''