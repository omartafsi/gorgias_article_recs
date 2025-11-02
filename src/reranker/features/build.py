import numpy as np
import pandas as pd
from reranker.utils.features_utils import build_feature_row
from reranker.utils.embeddings import mean_pool_query


def build_training_examples(
        fb_df: pd.DataFrame,
        article_emb_df: pd.DataFrame,
        article_emb_matrix: np.ndarray,
        n_random_negatives: int = 3,
        rng_seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build positive and negative examples for reranker training.
    Each feedback row contributes:
      - 1 positive example (correct article)
      - 1 negative example (recommended article if correct != recommended)
      - several negative examples (random within the same help center)
    """
    id_to_index = {id_: i for i, id_ in enumerate(article_emb_df["id"])}
    rng = np.random.default_rng(rng_seed)
    rows, labels = [], []
    help_centers = article_emb_df["help_center_id"].to_numpy()

    for _, row in fb_df.iterrows():
        query = row["query_embeddings_parsed"]
        if not query:
            continue

        help_center_id = row["help_center_id"]
        correct_article_id = row["correct_article_id"]
        recommended_article_id = row["recommended_article_id"]

        if correct_article_id not in id_to_index or recommended_article_id not in id_to_index:
            continue

        query_emb = mean_pool_query(query)

        # Add positive article
        correct_article_emb = article_emb_matrix[id_to_index[correct_article_id]]
        rows.append(build_feature_row(query_emb, correct_article_emb, help_center_id, correct_article_id))
        labels.append(1)

        # Add recommended article as negative if different from correct
        if recommended_article_id != correct_article_id:
            recommended_article_index = id_to_index[recommended_article_id]
            recommended_article_emb = article_emb_matrix[recommended_article_index]
            rows.append(build_feature_row(query_emb, recommended_article_emb, help_center_id, recommended_article_id))
            labels.append(0)

        # Add negative random articles (within same help center)
        negative_indexes = np.where((help_centers == help_center_id) & (article_emb_df["id"] != correct_article_id))[0]
        if len(negative_indexes) == 0:
            continue

        for i in rng.choice(negative_indexes, size=min(n_random_negatives, len(negative_indexes)), replace=False):
            rows.append(
                build_feature_row(query_emb, article_emb_matrix[i], help_center_id, article_emb_df["id"].iat[i])
            )
            labels.append(0)

    return pd.DataFrame(rows), np.array(labels, dtype=np.int32)


def build_candidate_features(
        query_emb: np.ndarray,
        article_emb_df: pd.DataFrame,
        article_emb_matrix: np.ndarray,
        candidate_article_indices: np.ndarray,
        help_center_id: str,
) -> pd.DataFrame:
    """
    Build features for candidate ranking at inference or evaluation time.
    """
    feats = [
        build_feature_row(query_emb, article_emb_matrix[i], help_center_id, article_emb_df["id"].iat[i])
        for i in candidate_article_indices
    ]
    return pd.DataFrame(feats)


def build_candidate_features_inference(
        query_emb: np.ndarray,
        candidate_article_embeddings: list[np.ndarray],
        candidate_article_ids: list[str],
        help_center_id: str,
) -> pd.DataFrame:
    """
    Build features for reranking at inference time using preloaded embeddings.
    This version does NOT rely on emb_df or candidate_indices
    """
    feats = [
        build_feature_row(query_emb,
                          np.array(article_embedding, dtype=np.float32), help_center_id, article_id)
        for article_embedding, article_id in zip(candidate_article_embeddings, candidate_article_ids)
    ]
    return pd.DataFrame(feats)
