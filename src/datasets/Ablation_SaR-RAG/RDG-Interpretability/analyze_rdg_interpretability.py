# -*- coding: utf-8 -*-
"""
RDG post-hoc interpretability analysis (spaCy version).

Merge dev sets of trivia / nq / hotpot, join question text with the useRAG label,
extract surface + linguistic features (with spaCy NER / POS), then run statistics
on:
    (a) full merged population
    (b) a fixed-seed random sample of 100 useRAG=0 + 100 useRAG=1

USAGE
-----
    # default paths assume the three subfolders sit next to this script
    python analyze_rdg_interpretability.py \
        --root /path/to/ESWA_revision \
        --spacy-model en_core_web_sm \
        --seed 42 \
        --sample-size 100

If en_core_web_sm is not installed, run:
    python -m spacy download en_core_web_sm

OUTPUTS  (written under --root)
-------------------------------
    features_full.csv          per-question features + useRAG + dataset
    stats_full.csv             population-level statistics per feature
    stats_sample.csv           sample-level statistics per feature
    sample.csv                 the sampled rows (with all features)
    summary.txt                human-readable summary

FEATURES
--------
Continuous:
    word_count            spaCy token count (excluding pure-space tokens)
    char_count            len(question.strip())
    digit_char_count      number of digit characters
    number_token_count    spaCy tokens with like_num=True
    entity_count          spaCy NER entity count
    person_count          # PERSON entities
    org_gpe_loc_count     # ORG + GPE + LOC + FAC entities
    date_time_count       # DATE + TIME entities
    cardinal_count        # CARDINAL + QUANTITY + PERCENT + MONEY entities
    noun_chunk_count      number of noun chunks
    verb_count            tokens with POS == VERB
    propn_count           tokens with POS == PROPN

Binary:
    starts_with_{wh}      where wh in {why, how, what, who, when, where, which}
    is_definition         "what is/are/was/were ...", "define ...", "what does ... mean"
    is_yesno              first token is be/have/do/modal aux
    is_comparison         contains 'same', ' or ', 'compare', 'between', 'both', ' vs '
    has_quote             contains a quote character
    has_person_entity     1 if any PERSON entity
    has_date_entity       1 if any DATE entity
    has_number_entity     1 if any CARDINAL/QUANTITY/PERCENT/MONEY entity

STAT TESTS
----------
    continuous : Welch's t-test + Mann-Whitney U + Cohen's d
    binary     : chi-square (Yates) on 2x2, plus absolute prop diff
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy import stats


WH_WORDS = ["why", "how", "what", "who", "when", "where", "which"]
YESNO_STARTERS = {
    "is", "are", "was", "were", "am",
    "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "has", "have", "had",
    "may", "might", "must",
}
COMPARISON_CUES = ["same", " or ", "compare", "between", "both", " vs ", " vs."]

PERSON_LABELS = {"PERSON"}
ORG_LOC_LABELS = {"ORG", "GPE", "LOC", "FAC", "NORP"}
DATETIME_LABELS = {"DATE", "TIME"}
NUMBER_LABELS = {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}


def load_trivia(root: Path):
    qmap = {}
    with open(root / "trivia" / "trivia_qa_dev_cleaned.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qmap[r["question_id"]] = r["question"]
    rows = []
    with open(root / "trivia" / "trivia_qa_dev_results_cleaned.jsonl",
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = r["id"]
            if qid in qmap:
                rows.append({
                    "id": qid,
                    "question": qmap[qid],
                    "useRAG": int(r["useRAG"]),
                    "dataset": "trivia",
                })
    return rows


def load_nq(root: Path):
    rows = []
    with open(root / "nq" / "NQ-open_dev_standard.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "id": r["id"],
                "question": r["question"],
                "useRAG": int(r["useRAG"]),
                "dataset": "nq",
            })
    return rows


def load_hotpot(root: Path):
    qmap = {}
    with open(root / "hotpot" / "hotpot_dev_dedup.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qmap[r["_id"]] = r["question"]
    rows = []
    with open(root / "hotpot" / "hotpot_dev_results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = r["id"]
            if qid in qmap:
                rows.append({
                    "id": qid,
                    "question": qmap[qid],
                    "useRAG": int(r["useRAG"]),
                    "dataset": "hotpot",
                })
    return rows


def rule_features(q: str) -> dict:
    q_strip = q.strip()
    tokens = q_strip.split()
    first = tokens[0].lower().strip(",.?!:;\"'") if tokens else ""
    lower = q_strip.lower()

    feats = {
        "char_count": len(q_strip),
        "digit_char_count": sum(c.isdigit() for c in q_strip),
        "has_quote": int(any(c in q_strip for c in "\"'`")),
    }
    for w in WH_WORDS:
        feats[f"starts_with_{w}"] = int(first == w)

    is_def = 0
    if lower.startswith(("what is ", "what are ", "what was ", "what were ")):
        is_def = 1
    if lower.startswith(("define ", "definition of ")):
        is_def = 1
    if lower.startswith("what does ") and " mean" in lower:
        is_def = 1
    feats["is_definition"] = is_def

    feats["is_yesno"] = int(first in YESNO_STARTERS)
    feats["is_comparison"] = int(any(c in lower for c in COMPARISON_CUES))
    return feats


def spacy_features(doc) -> dict:
    word_count = sum(1 for t in doc if not t.is_space)
    number_token_count = sum(1 for t in doc if t.like_num)
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    propn_count = sum(1 for t in doc if t.pos_ == "PROPN")
    noun_chunk_count = sum(1 for _ in doc.noun_chunks)

    person = org = dt = num = 0
    for ent in doc.ents:
        if ent.label_ in PERSON_LABELS:
            person += 1
        elif ent.label_ in ORG_LOC_LABELS:
            org += 1
        elif ent.label_ in DATETIME_LABELS:
            dt += 1
        elif ent.label_ in NUMBER_LABELS:
            num += 1
    entity_count = len(doc.ents)

    return {
        "word_count": word_count,
        "number_token_count": number_token_count,
        "verb_count": verb_count,
        "propn_count": propn_count,
        "noun_chunk_count": noun_chunk_count,
        "entity_count": entity_count,
        "person_count": person,
        "org_gpe_loc_count": org,
        "date_time_count": dt,
        "cardinal_count": num,
        "has_person_entity": int(person > 0),
        "has_date_entity": int(dt > 0),
        "has_number_entity": int(num > 0),
    }


CONTINUOUS = [
    "word_count", "char_count", "digit_char_count", "number_token_count",
    "entity_count", "person_count", "org_gpe_loc_count",
    "date_time_count", "cardinal_count",
    "noun_chunk_count", "verb_count", "propn_count",
]
BINARY = [
    "has_quote", "is_definition", "is_yesno", "is_comparison",
    "has_person_entity", "has_date_entity", "has_number_entity",
] + [f"starts_with_{w}" for w in WH_WORDS]


def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df.useRAG == 1]
    neg = df[df.useRAG == 0]
    rows = []

    for f in CONTINUOUS:
        a, b = neg[f].to_numpy(), pos[f].to_numpy()
        if len(a) < 2 or len(b) < 2:
            t_p = u_p = float("nan")
        else:
            _, t_p = stats.ttest_ind(a, b, equal_var=False)
            try:
                _, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                u_p = float("nan")
        var_a = a.var(ddof=1) if len(a) > 1 else 0.0
        var_b = b.var(ddof=1) if len(b) > 1 else 0.0
        pooled = np.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0.0
        d = (b.mean() - a.mean()) / pooled if pooled > 0 else float("nan")
        rows.append({
            "feature": f, "kind": "continuous",
            "neg_n": len(a), "pos_n": len(b),
            "neg_mean": round(float(a.mean()), 4),
            "pos_mean": round(float(b.mean()), 4),
            "neg_std": round(float(np.sqrt(var_a)), 4),
            "pos_std": round(float(np.sqrt(var_b)), 4),
            "diff": round(float(b.mean() - a.mean()), 4),
            "cohen_d": None if d != d else round(float(d), 4),
            "welch_p": f"{t_p:.3g}",
            "mwu_p": f"{u_p:.3g}",
            "chi2_p": "",
        })

    for f in BINARY:
        a, b = neg[f].to_numpy(), pos[f].to_numpy()
        a1, a0 = int(a.sum()), int(len(a) - a.sum())
        b1, b0 = int(b.sum()), int(len(b) - b.sum())
        table = np.array([[a1, a0], [b1, b0]])
        if table.min() < 0 or table.sum() == 0:
            p = float("nan")
        else:
            try:
                _, p, _, _ = stats.chi2_contingency(table, correction=True)
            except ValueError:
                p = float("nan")
        rows.append({
            "feature": f, "kind": "binary",
            "neg_n": len(a), "pos_n": len(b),
            "neg_mean": round(float(a.mean()), 4),
            "pos_mean": round(float(b.mean()), 4),
            "neg_std": "", "pos_std": "",
            "diff": round(float(b.mean() - a.mean()), 4),
            "cohen_d": "",
            "welch_p": "", "mwu_p": "",
            "chi2_p": f"{p:.3g}",
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                   help="folder containing trivia/, nq/, hotpot/ subdirs")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=100,
                   help="N per class (will draw N from useRAG=0 and N from useRAG=1)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-process", type=int, default=1,
                   help="spaCy nlp.pipe n_process; >1 only on Linux/macOS")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    print(f"[info] root = {root}")
    print(f"[info] loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model, disable=["lemmatizer"])

    rows = load_trivia(root) + load_nq(root) + load_hotpot(root)
    print(f"[info] joined questions: {len(rows)}")

    df = pd.DataFrame(rows)
    print("[info] useRAG x dataset:")
    print(df.groupby(["dataset", "useRAG"]).size().unstack(fill_value=0))

    rule_feats = [rule_features(q) for q in df.question]

    print(f"[info] running spaCy pipeline (batch={args.batch_size}, "
          f"n_process={args.n_process}) ...")
    sp_feats = []
    for doc in nlp.pipe(df.question.tolist(),
                        batch_size=args.batch_size,
                        n_process=args.n_process):
        sp_feats.append(spacy_features(doc))

    feat_df = pd.concat(
        [pd.DataFrame(rule_feats), pd.DataFrame(sp_feats)], axis=1)
    full = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    full.to_csv(root / "features_full.csv", index=False, encoding="utf-8-sig")
    print(f"[ok] wrote features_full.csv  ({len(full)} rows)")

    full_stats = stats_table(full)
    full_stats.to_csv(root / "stats_full.csv", index=False, encoding="utf-8-sig")
    print("\n=== FULL POPULATION STATS ===")
    print(full_stats.to_string(index=False))

    rng = random.Random(args.seed)
    neg_pool = full.index[full.useRAG == 0].tolist()
    pos_pool = full.index[full.useRAG == 1].tolist()
    rng.shuffle(neg_pool)
    rng.shuffle(pos_pool)
    n = args.sample_size
    if len(neg_pool) < n or len(pos_pool) < n:
        raise ValueError(
            f"Not enough samples: useRAG=0 has {len(neg_pool)}, "
            f"useRAG=1 has {len(pos_pool)}, requested {n} each")
    sample = full.loc[neg_pool[:n] + pos_pool[:n]].reset_index(drop=True)
    sample.to_csv(root / "sample.csv", index=False, encoding="utf-8-sig")

    sample_stats = stats_table(sample)
    sample_stats.to_csv(root / "stats_sample.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== {n}+{n} SAMPLE STATS (seed={args.seed}) ===")
    print(sample_stats.to_string(index=False))

    with open(root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"root = {root}\n")
        f.write(f"spacy_model = {args.spacy_model}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f"sample_size_per_class = {n}\n\n")
        f.write(f"merged questions: {len(full)}\n")
        f.write(f"useRAG=0: {(full.useRAG == 0).sum()}\n")
        f.write(f"useRAG=1: {(full.useRAG == 1).sum()}\n\n")
        f.write("--- per-dataset useRAG distribution ---\n")
        f.write(str(df.groupby(["dataset", "useRAG"]).size().unstack(fill_value=0)))
        f.write("\n\n--- FULL STATS ---\n")
        f.write(full_stats.to_string(index=False))
        f.write("\n\n--- SAMPLE STATS ---\n")
        f.write(sample_stats.to_string(index=False))
    print("\n[ok] wrote summary.txt")


if __name__ == "__main__":
    main()
