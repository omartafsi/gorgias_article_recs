import numpy as np


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def hellinger_similarity(a, b):
    # Clip to avoid sqrt of negative values
    a = np.clip(a, 1e-9, None)
    b = np.clip(b, 1e-9, None)
    sim = 1.0 - (np.sqrt(0.5 * np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)))
    return float(sim)


def build_feature_row(query_embedding: np.ndarray, article_embedding: np.ndarray, help_center_id: str,
                      article_id: str) -> dict:
    """Compute all numeric and categorical features for a query–article pair with shape (384,)
     """

    return {
        "cosine": cosine(query_embedding, article_embedding),
        "hellinger": hellinger_similarity(query_embedding, article_embedding),
        "help_center_id": help_center_id,
        "article_id": article_id
    }
