# src/metrics.py
from rouge_score import rouge_scorer
import math

def simple_bleu(pred, ref, n_gram=4):
    """
    Простая и быстрая реализация BLEU без NLTK, совместима с Python 3.12.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []

    for i in range(1, n_gram + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)]
        ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)]
        if len(pred_ngrams) == 0:
            precisions.append(0)
            continue
        overlap = sum(1 for ng in pred_ngrams if ng in ref_ngrams)
        precisions.append(overlap / len(pred_ngrams))

    # Геометрическое среднее
    score = math.exp(sum(w * math.log(p + 1e-9) for w, p in zip(weights, precisions)))

    # Brevity penalty (наказание за слишком короткие)
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    bp = math.exp(1 - ref_len / pred_len) if pred_len < ref_len else 1.0

    return bp * score


def compute_bleu_rouge(preds, refs):
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    for pred, ref in zip(preds, refs):
        bleu_scores.append(simple_bleu(pred, ref))
        r = scorer.score(ref, pred)
        rouge_scores["rouge1"].append(r["rouge1"].fmeasure)
        rouge_scores["rougeL"].append(r["rougeL"].fmeasure)

    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
        "ROUGEL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]),
    }


def approx_spice(preds, refs):
    """Упрощённая имитация SPICE (по смыслу словарного пересечения)."""
    scores = []
    for p, r in zip(preds, refs):
        p_words = set(p.lower().split())
        r_words = set(r.lower().split())
        scores.append(len(p_words & r_words) / max(len(r_words), 1))
    return sum(scores) / len(scores) if scores else 0.0


def compute_metrics(preds, refs):
    base_metrics = compute_bleu_rouge(preds, refs)
    base_metrics["SPICE"] = approx_spice(preds, refs)
    return base_metrics
