from typing import Sequence, Dict


def hit_at_k(ranked_ids: Sequence[str], target_id: str, k: int = 1) -> float:
    return float(target_id in list(ranked_ids)[:k])


def mrr(ranked_ids: Sequence[str], target_id: str) -> float:
    for i, id_ in enumerate(ranked_ids, start=1):
        if id_ == target_id:
            return 1.0 / i
    return 0.0


def compute_metrics(cosine_ids, model_ids, correct_id) -> Dict[str, float]:
    """Compute comparison metrics between cosine and reranker outputs."""
    return {
        "hit1_cos": hit_at_k(cosine_ids, correct_id, 1),
        "hit3_cos": hit_at_k(cosine_ids, correct_id, 3),
        "mrr_cos": mrr(cosine_ids, correct_id),
        "hit1_model": hit_at_k(model_ids, correct_id, 1),
        "hit3_model": hit_at_k(model_ids, correct_id, 3),
        "mrr_model": mrr(model_ids, correct_id),
    }
