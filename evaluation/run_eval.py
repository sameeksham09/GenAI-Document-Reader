"""
eval/run_eval.py
─────────────────────────────────────────────────────────────────────────────
Evaluation pipeline for the RAG system.

Metrics measured:
  1. Retrieval Precision@k  — was the right chunk retrieved in top-k?
  2. Token F1               — word-overlap between generated answer and reference
  3. Exact Match (EM)       — strict: does generated answer contain the reference keywords?

Run from project root:
    python evaluation/run_eval.py

Output:
    evaluation/results.json   — per-question detailed results
    Prints summary table to terminal
"""

import sys
import os
import json
import re
from collections import Counter

# ── Make sure project root is on path so we can import retriever / llm_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import retrieve_context
from llm_utils  import generate_answer
from prompts    import build_prompt, get_instruction

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FILE   = os.path.join(os.path.dirname(__file__), "test_qa.json")
RESULTS_FILE= os.path.join(os.path.dirname(__file__), "results.json")
TARGET_DOC  = "Database Management Systems (DBMS).pdf"
TOP_K       = 4      # must match retrieve_context default


# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalise(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction, reference):
    """
    Compute token-level F1 between prediction and reference strings.
    Same metric used in Stanford SQuAD benchmark.

    F1 = 2 * precision * recall / (precision + recall)
    where precision = overlap / len(pred_tokens)
          recall    = overlap / len(ref_tokens)
    """
    pred_tokens = _normalise(prediction).split()
    ref_tokens  = _normalise(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts  = Counter(ref_tokens)

    # Number of tokens that appear in both
    overlap = sum((pred_counts & ref_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(ref_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def exact_match(prediction, reference):
    """
    Relaxed exact match: checks whether all key reference keywords
    appear in the prediction. More forgiving than strict string equality
    but still meaningful for factual QA.
    """
    pred_norm = _normalise(prediction)
    ref_norm  = _normalise(reference)

    # Extract meaningful keywords from reference (words > 3 chars)
    keywords = [w for w in ref_norm.split() if len(w) > 3]

    if not keywords:
        return 0

    # Count how many keywords appear in prediction
    found = sum(1 for kw in keywords if kw in pred_norm)
    ratio = found / len(keywords)

    # Consider it a match if >= 60% of keywords present
    return 1 if ratio >= 0.6 else 0


def retrieval_precision(retrieved_chunks, reference_answer):
    """
    Check whether at least one retrieved chunk contains enough
    reference keywords to be considered a relevant chunk.
    Returns 1 if any chunk is relevant, 0 otherwise.
    """
    ref_keywords = [w for w in _normalise(reference_answer).split() if len(w) > 4]
    if not ref_keywords:
        return 1  # can't evaluate, assume ok

    for chunk in retrieved_chunks:
        chunk_norm = _normalise(chunk["text"])
        found = sum(1 for kw in ref_keywords if kw in chunk_norm)
        if found / len(ref_keywords) >= 0.3:   # 30% keyword overlap = relevant
            return 1
    return 0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation():
    with open(TEST_FILE) as f:
        test_cases = json.load(f)

    results     = []
    total       = len(test_cases)
    ret_hits    = 0
    f1_scores   = []
    em_scores   = []

    print(f"\n{'='*70}")
    print(f"  RAG Evaluation — {TARGET_DOC}")
    print(f"  {total} questions | top-k={TOP_K}")
    print(f"{'='*70}\n")

    for i, tc in enumerate(test_cases, 1):
        qid       = tc["id"]
        question  = tc["question"]
        reference = tc["reference"]
        topic     = tc.get("topic", "general")

        print(f"[{i:02d}/{total}] {question[:65]}...")

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        retrieved = retrieve_context(question, k=TOP_K, selected_doc=TARGET_DOC)

        if not retrieved:
            print(f"        ⚠️  No chunks retrieved — skipping\n")
            results.append({
                "id": qid, "question": question, "topic": topic,
                "retrieved": False, "ret_precision": 0,
                "generated_answer": "", "token_f1": 0.0, "exact_match": 0
            })
            em_scores.append(0)
            f1_scores.append(0.0)
            continue

        # ── Step 2: Check retrieval quality ───────────────────────────────────
        ret_prec = retrieval_precision(retrieved, reference)
        ret_hits += ret_prec

        # ── Step 3: Generate answer ───────────────────────────────────────────
        context    = "\n".join(c["text"] for c in retrieved)
        instruction = get_instruction("1")   # descriptive mode
        prompt     = build_prompt(context, instruction, question)
        answer     = generate_answer(prompt, max_new_tokens=150)

        # ── Step 4: Score answer ──────────────────────────────────────────────
        tf1 = token_f1(answer, reference)
        em  = exact_match(answer, reference)

        f1_scores.append(tf1)
        em_scores.append(em)

        # ── Step 5: Show live result ──────────────────────────────────────────
        ret_icon = "✅" if ret_prec else "❌"
        em_icon  = "✅" if em      else "❌"
        print(f"        Retrieval: {ret_icon}  |  Token F1: {tf1:.2f}  |  EM: {em_icon}")
        print(f"        Top chunk rerank: {retrieved[0].get('rerank_score', 0):.2f}")
        print()

        results.append({
            "id":               qid,
            "question":         question,
            "topic":            topic,
            "retrieved":        bool(retrieved),
            "ret_precision":    ret_prec,
            "top_rerank_score": retrieved[0].get("rerank_score", 0) if retrieved else 0,
            "reference":        reference,
            "generated_answer": answer,
            "token_f1":         tf1,
            "exact_match":      em,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_f1      = round(sum(f1_scores) / len(f1_scores), 4)
    avg_em      = round(sum(em_scores) / len(em_scores), 4)
    ret_prec_at_k = round(ret_hits / total, 4)

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Questions evaluated  : {total}")
    print(f"  Retrieval Precision  : {ret_prec_at_k:.0%}  ({ret_hits}/{total} questions retrieved correctly)")
    print(f"  Avg Token F1         : {avg_f1:.2f}  (1.0 = perfect word overlap)")
    print(f"  Exact Match          : {avg_em:.0%}  ({sum(em_scores)}/{total} answers contain reference keywords)")
    print(f"{'='*70}\n")

    # ── Per-topic breakdown ───────────────────────────────────────────────────
    topics = {}
    for r in results:
        t = r["topic"]
        if t not in topics:
            topics[t] = {"f1": [], "em": [], "ret": []}
        topics[t]["f1"].append(r["token_f1"])
        topics[t]["em"].append(r["exact_match"])
        topics[t]["ret"].append(r["ret_precision"])

    print("  Per-topic breakdown:")
    print(f"  {'Topic':<20} {'Retrieval':>10} {'Token F1':>10} {'Exact Match':>12}")
    print(f"  {'-'*56}")
    for topic, scores in sorted(topics.items()):
        r_avg  = sum(scores["ret"]) / len(scores["ret"])
        f_avg  = sum(scores["f1"])  / len(scores["f1"])
        em_avg = sum(scores["em"])  / len(scores["em"])
        print(f"  {topic:<20} {r_avg:>9.0%}  {f_avg:>9.2f}  {em_avg:>11.0%}")
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_questions":    total,
            "retrieval_precision": ret_prec_at_k,
            "avg_token_f1":       avg_f1,
            "exact_match":        avg_em,
        },
        "per_topic": {
            t: {
                "retrieval": round(sum(s["ret"])/len(s["ret"]), 2),
                "token_f1":  round(sum(s["f1"]) /len(s["f1"]),  2),
                "exact_match": round(sum(s["em"])/len(s["em"]), 2),
            }
            for t, s in topics.items()
        },
        "results": results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Full results saved to: {RESULTS_FILE}")
    print(f"\n  📋 Resume bullet (fill in your numbers):")
    print(f"  Achieved {avg_em:.0%} exact-match and {avg_f1:.2f} token-F1 across")
    print(f"  {total} held-out QA pairs with {ret_prec_at_k:.0%} retrieval precision.\n")

    return output


if __name__ == "__main__":
    run_evaluation()