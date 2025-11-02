import numpy as np


def mean_pool_query(chunks):
    """Average-pools multiple query embeddings into a single vector."""
    if chunks is None or len(chunks) == 0:
        return None
    a = np.asarray(chunks, dtype=np.float32)
    return a.mean(axis=0)
